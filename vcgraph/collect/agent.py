"""Data Collection Agent (LangGraph + Crawl4AI -> MongoDB -> entity resolution -> Neo4j).

    plan ─▶ yc ─▶ (no companies? END) ─▶ sec ─▶ opencorporates ─▶ resolve ─▶ load_kg ─▶ END

* yc            : Crawl4AI renders the YC directory, discovers company pages and crawls them;
                  an LLM turns each page's markdown into a CompanyProfile (no hand-written selectors).
                  `yc_mode="json"` uses the community JSON mirror instead (fast, no browser).
* sec           : EDGAR full-text search for Form D filings per company, XML parsed into filings.
* opencorporates: legal-entity registrations (skipped when no API token).
* resolve       : merges duplicates across the three sources (see resolve.py).
* load_kg       : builds the Neo4j graph + vector embeddings.
Every raw payload lands in Mongo `raw_pages` first; normalized records in `records`.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..jsonparse import parser_stats
from ..llm import Usage
from ..storage import mongo
from . import sources
from .extract import extract_company, profile_to_record
from .resolve import resolve


class CState(TypedDict, total=False):
    batches: list[str]
    limit: int
    yc_mode: str           # "crawl" | "json" | "hybrid"
    skip_sec: bool
    skip_kg: bool
    use_llm_er: bool
    company_records: list[dict]
    filing_records: list[dict]
    legal_records: list[dict]
    entities: list[dict]
    log: list[str]
    stats: dict[str, Any]
    usage: Any


def _log(s: CState, msg: str) -> list[str]:
    print(f"[collect] {msg}")
    return [*s.get("log", []), msg]


async def n_plan(s: CState) -> CState:
    return {"log": _log(s, f"plan: batches={s['batches']} limit/batch={s.get('limit', 20)} mode={s.get('yc_mode', 'crawl')}"),
            "stats": {}, "usage": Usage(), "company_records": [], "filing_records": [], "legal_records": []}


async def n_yc(s: CState) -> CState:
    t0 = time.perf_counter()
    mode, limit = s.get("yc_mode", "crawl"), s.get("limit", 20)
    recs: list[dict] = []
    for batch in s["batches"]:
        if mode in ("json", "hybrid"):
            rows = await sources.fetch_yc_oss(batch, limit)
            batch_recs = [sources.yc_oss_to_record(r, batch) for r in rows]
            if mode == "hybrid":
                await _add_founders(batch_recs, s["usage"])
            recs += batch_recs
        else:  # "crawl": discover + extract everything from www.ycombinator.com
            urls = await sources.crawl_yc_directory(sources.normalize_batch(batch), limit)
            pages = await sources.crawl_pages(urls)
            for pg in pages:
                prof = await asyncio.to_thread(extract_company, pg["markdown"], pg["url"], None, s["usage"])
                if prof:
                    rec = profile_to_record(prof, pg["url"])
                    rec["data"]["batch"] = sources.normalize_batch(rec["data"].get("batch") or batch)
                    recs.append(rec)
    mongo.save_records(recs)
    n_f = sum(1 for r in recs if r["data"].get("founders"))
    stats = {**s.get("stats", {}), "yc_companies": len(recs), "yc_with_founders": n_f,
             "yc_time_s": round(time.perf_counter() - t0, 2)}
    return {"company_records": recs, "stats": stats,
            "log": _log(s, f"yc[{mode}]: {len(recs)} company records ({n_f} with founders)")}


async def _add_founders(recs: list[dict], usage) -> None:
    """hybrid mode: JSON mirror gives clean metadata; crawl each YC company page for founders."""
    by_url = {r["data"]["yc_url"]: r for r in recs if r["data"].get("yc_url")}
    pages = await sources.crawl_pages(list(by_url))
    for pg in pages:
        prof = await asyncio.to_thread(extract_company, pg["markdown"], pg["url"], None, usage)
        rec = by_url.get(pg["url"])
        if prof and rec:
            rec["data"]["founders"] = [f.model_dump() for f in prof.founders]
            rec["data"]["investors"] = prof.investors
            if not rec["data"].get("founded_year"):
                rec["data"]["founded_year"] = prof.founded_year


async def n_sec(s: CState) -> CState:
    if s.get("skip_sec"):
        return {}
    t0 = time.perf_counter()
    out: list[dict] = []
    async with sources._sec_client() as c:
        for rec in s["company_records"]:
            try:
                for hit in await sources.sec_search_form_d(c, rec["name"]):
                    fd = await sources.sec_fetch_form_d(c, hit)
                    if (fd and sources.issuer_matches(fd.get("issuer_name"), rec["name"])
                            and sources.filing_plausible(fd, rec["data"].get("batch"))):
                        out.append(sources.form_d_to_record(fd, rec["name"]))
            except Exception as e:  # noqa: BLE001
                print(f"  sec error for {rec['name']}: {e}")
    mongo.save_records(out)
    stats = {**s["stats"], "sec_filings": len(out), "sec_time_s": round(time.perf_counter() - t0, 2)}
    return {"filing_records": out, "stats": stats, "log": _log(s, f"sec: {len(out)} Form D filings")}


async def n_oc(s: CState) -> CState:
    import httpx

    out: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as c:
        for rec in s["company_records"]:
            try:
                out += await sources.opencorporates_search(c, rec["name"])
            except Exception as e:  # noqa: BLE001
                print(f"  opencorporates error for {rec['name']}: {e}")
    mongo.save_records(out)
    return {"legal_records": out, "stats": {**s["stats"], "oc_entities": len(out)},
            "log": _log(s, f"opencorporates: {len(out)} legal entities")}


async def n_resolve(s: CState) -> CState:
    allrec = s["company_records"] + s.get("filing_records", []) + s.get("legal_records", [])
    ents, st = await asyncio.to_thread(resolve, allrec, s.get("use_llm_er", False))
    # drop clusters that came only from SEC/OC fuzzy hits without a matching company
    ents = [e for e in ents if any(src in ("yc", "demo") for src in e["sources"])]
    mongo.save_entities(ents)
    return {"entities": ents, "stats": {**s["stats"], "er": st},
            "log": _log(s, f"resolve: {st['records']} records -> {len(ents)} companies ({st['merges']} merges)")}


async def n_load(s: CState) -> CState:
    if s.get("skip_kg"):
        return {}
    from ..kg.builder import build_graph
    from ..storage.graph import Neo4jGraph

    g = Neo4jGraph()
    try:
        st = await asyncio.to_thread(build_graph, s["entities"], g)
    finally:
        g.close()
    stats = {**s["stats"], "kg": st, "parser": parser_stats(), "llm": s["usage"].summary()}
    return {"stats": stats, "log": _log(s, f"kg: {st}")}


def build_collect_agent():
    sg = StateGraph(CState)
    for name, fn in [("plan", n_plan), ("yc", n_yc), ("sec", n_sec), ("opencorporates", n_oc),
                     ("resolve", n_resolve), ("load_kg", n_load)]:
        sg.add_node(name, fn)
    sg.add_edge(START, "plan")
    sg.add_edge("plan", "yc")
    sg.add_conditional_edges("yc", lambda s: "sec" if s.get("company_records") else END,
                             {"sec": "sec", END: END})
    sg.add_edge("sec", "opencorporates")
    sg.add_edge("opencorporates", "resolve")
    sg.add_edge("resolve", "load_kg")
    sg.add_edge("load_kg", END)
    return sg.compile()


async def run_collection(batches: list[str], limit: int = 20, yc_mode: str = "crawl",
                         skip_sec: bool = False, skip_kg: bool = False, use_llm_er: bool = False) -> dict:
    app = build_collect_agent()
    out = await app.ainvoke({"batches": batches, "limit": limit, "yc_mode": yc_mode, "skip_sec": skip_sec,
                             "skip_kg": skip_kg, "use_llm_er": use_llm_er})
    return out.get("stats", {})
