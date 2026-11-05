"""Investor extraction from public funding news (data collection agent, part 2).

Pipeline
  1. harvest   : funding-related links from the "Latest News" section of each company's YC page
                 (already crawled and stored in Mongo raw_pages -> the company is known, no name ambiguity).
                 Optional: Google News RSS for companies whose YC page lists no funding news.
  2. fetch     : open each article with Crawl4AI (paywalled pages fall back to the headline only).
  3. extract   : LLM -> FundingEvent {about_company, round, amount_usd, date, lead_investors, investors},
                 given the company's one-liner so it can reject articles about a same-named company.
  4. normalize : investor names -> canonical key ("a16z" == "Andreessen Horowitz").
  5. merge     : funding events -> entity["investors"] in Mongo, ready for the KG builder.
Every investor edge keeps its source URL and date, so it can be audited and compared with PitchBook.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Optional


from pydantic import BaseModel, Field, model_validator

from ..jsonparse import parse_model
from ..llm import Usage, chat
from ..storage import mongo
from .resolve import normalize_name, slugify

# ------------------------------------------------------------------ 1. harvest
FUNDING_RE = re.compile(
    r"rais(e|es|ed|ing)|funding|fundrais|\bseed\b|pre-seed|series[\s\-_]?[a-h]\b|investment|invests?\b|"
    r"backed|led\s+by|valuation|\bround\b|financing", re.I)
SKIP_DOMAINS = ("youtube.com", "youtu.be", "twitter.com", "x.com", "linkedin.com", "instagram.com",
                "tiktok.com", "ycombinator.com")
LINK_RE = re.compile(r"^\[(?P<title>.+?)\]\((?P<url>https?://[^)\s]+)\)\s*$")
DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$")


def parse_yc_news(markdown: str) -> list[dict]:
    """[{title, url, date}] from the 'Latest News' block of a YC company page (markdown)."""
    i = markdown.find("Latest News")
    if i < 0:
        return []
    out: list[dict] = []
    for line in markdown[i + len("Latest News"):].splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINK_RE.match(line)
        if m:
            out.append({"title": m["title"].strip(), "url": m["url"], "date": None})
        elif DATE_RE.match(line) and out and out[-1]["date"] is None:
            out[-1]["date"] = line
        else:
            break                                    # next section ("Company Launches", "YC Photos", ...)
    return out


def is_funding_news(item: dict) -> bool:
    host = urllib.parse.urlparse(item["url"]).netloc.lower()
    if any(d in host for d in SKIP_DOMAINS):
        return False
    path = urllib.parse.urlparse(item["url"]).path.replace("-", " ").replace("_", " ")
    return bool(FUNDING_RE.search(item["title"]) or FUNDING_RE.search(path))


def harvest_yc_news(entities: list[dict], per_company: int = 4) -> dict[str, list[dict]]:
    """entity_id -> funding-news candidates taken from the company's own YC page."""
    url_of = {e["attrs"].get("yc_url"): e["entity_id"] for e in entities if (e.get("attrs") or {}).get("yc_url")}
    out: dict[str, list[dict]] = {}
    for doc in mongo.db().raw_pages.find({"source": "yc", "url": {"$in": list(url_of)}},
                                         {"url": 1, "meta.markdown": 1}):
        items = [n for n in parse_yc_news(doc.get("meta", {}).get("markdown", "")) if is_funding_news(n)]
        if items:
            out[url_of[doc["url"]]] = [{**n, "via": "yc_page"} for n in items[:per_company]]
    return out


async def rss_candidates(name: str, after: str = "2022-01-01", max_items: int = 3) -> list[dict]:
    """Google News RSS fallback (name-ambiguous: the extractor must confirm the company)."""
    import httpx

    q = f'"{name}" (raises OR raised OR funding OR seed OR "Series A" OR "Series B") after:{after}'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        r = await c.get(url)
    if r.status_code != 200:
        return []
    out = []
    for it in ET.fromstring(r.text).findall(".//item"):
        title = it.findtext("title") or ""
        if name.lower() in title.lower() and FUNDING_RE.search(title):
            out.append({"title": title, "url": it.findtext("link"), "date": it.findtext("pubDate"), "via": "rss"})
        if len(out) >= max_items:
            break
    return out


# ------------------------------------------------------------------ 3. extract
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
         "b": 1e9, "bn": 1e9, "billion": 1e9}


def parse_amount(v):
    """24500000 | "24.5M" | "$24.5 million" | "€10m" | "1.2 billion" -> float USD-ish (currency not converted)."""
    if v is None or isinstance(v, (int, float)):
        return v
    m = re.search(r"([\d.,]+)\s*(k|thousand|mm|mn|m|million|bn|b|billion)?\b", str(v).lower())
    if not m:
        return None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return num * _MULT.get(m.group(2) or "", 1)


class FundingEvent(BaseModel):
    about_company: Optional[bool] = Field(None, description="True only if the text is about THIS company")
    is_funding: Optional[bool] = Field(None, description="True if it reports a financing round of this company")
    round: Optional[str] = None
    amount_usd: Optional[float] = None
    announced_date: Optional[str] = None
    lead_investors: list[str] = Field(default_factory=list)
    investors: list[str] = Field(default_factory=list, description="ALL participating investors incl. leads")

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, v):
        """Small local models return many shapes: nested objects, null lists, '$24.5M', dicts per investor."""
        if isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), dict):
            v = next(iter(v.values()))                                 # {"FundingEvent": {...}}
        if not isinstance(v, dict):
            return v
        v = {str(k).strip().lower(): x for k, x in v.items()}
        for k in ("lead_investors", "investors"):
            x = v.get(k)
            if x is None:
                x = []
            elif isinstance(x, str):
                x = [p.strip() for p in re.split(r",|;| and ", x) if p.strip()]
            v[k] = [(i.get("name") or i.get("investor") or "") if isinstance(i, dict) else str(i) for i in x]
            v[k] = [i for i in v[k] if i and i.lower() not in ("none", "n/a", "unknown", "undisclosed")]
        v["amount_usd"] = parse_amount(v.get("amount_usd", v.get("amount")))
        for k in ("round", "announced_date"):
            if v.get(k) is not None and not isinstance(v[k], str):
                v[k] = str(v[k])
        for k in ("about_company", "is_funding"):
            if isinstance(v.get(k), str):
                t = v[k].strip().lower()
                v[k] = True if t in ("true", "yes", "y", "1") else False if t in ("false", "no", "n", "0") else None
        return v

_SYS = """You extract venture financing facts from a news article about a startup.
Use ONLY what the text states; never guess investors. Firms and named angel investors count as investors;
the company itself does not. If the text names no investors, return empty lists.
Return ONLY a JSON object with exactly these keys and types, e.g.:
{"about_company": true, "is_funding": true, "round": "Series A", "amount_usd": 24500000,
 "announced_date": "2025-04-25", "lead_investors": ["Benchmark"], "investors": ["Benchmark", "SV Angel"]}
about_company and is_funding MUST be the booleans true or false (not text).
about_company is false if the article is about a different company that merely has a similar name."""


def _clean(md: str, limit: int = 7000) -> str:
    md = re.sub(r"\(https?://[^)\s]+\)", "", md)
    md = re.sub(r"!\[[^\]]*\]", "", md)
    md = re.sub(r"\n{2,}", "\n", md)
    return md[:limit]


def extract_event(company: dict, item: dict, body: str | None, model: str | None = None,
                  usage: Usage | None = None) -> FundingEvent | None:
    a = company.get("attrs") or {}
    ctx = (f"COMPANY: {company['name']} | {a.get('one_liner') or ''} | website {a.get('website') or 'n/a'} "
           f"| YC {a.get('batch') or ''}\nHEADLINE: {item['title']}\nDATE: {item.get('date') or 'n/a'}\nURL: {item['url']}\n")
    ctx += f"ARTICLE:\n{_clean(body)}" if body else "ARTICLE: (not available, use the headline only)"
    text, _ = chat([{"role": "system", "content": _SYS}, {"role": "user", "content": ctx}],
                   model, json_mode=True, max_tokens=500, purpose="extract_funding", usage=usage)
    ev = parse_model(text, FundingEvent, kind="extract_funding")
    if ev is None:
        return None
    # the model answered with prose instead of a boolean: decide from the evidence we trust
    name = company["name"].lower()
    if ev.about_company is None:
        ev.about_company = item.get("via") == "yc_page" or name in item["title"].lower()
    if ev.is_funding is None:
        ev.is_funding = bool(FUNDING_RE.search(item["title"])) or ev.amount_usd is not None or bool(ev.investors)
    # small models tend to mark every investor as "lead": keep a lead only if the text says so
    text = item["title"] + "\n" + (_clean(body, 20000) if body else "")
    ev.lead_investors = [n for n in ev.lead_investors if _says_led(n, text)]
    ev.round = normalize_round(ev.round)
    return ev


_LED_BY = re.compile(r"\b(?:co-)?led\s+by\s+(.{0,160})", re.I)
_STOP = re.compile(r"\b(with|along|alongside|participation|joined|including|as well|plus)\b|[.;\n]", re.I)


def _says_led(name: str, text: str) -> bool:
    """True if the text explicitly attributes the lead to `name` ("led by X", "X led the round")."""
    t = text.lower()
    full = name.lower().split(" (")[0].strip()
    alts = [re.escape(full)]
    first = full.split()[0] if full.split() else ""
    if len(first) >= 4 and first not in ("capital", "ventures", "partners"):
        alts.append(re.escape(first))                    # "Benchmark Capital" ~ "Benchmark"
    n = re.compile(r"\b(" + "|".join(alts) + r")\b")
    for m in _LED_BY.finditer(t):                         # "led by A and B, with participation from C"
        seg = _STOP.split(m.group(1), maxsplit=1)[0]
        if n.search(seg):
            return True
    return bool(re.search(n.pattern + r"[^.\n]{0,40}\b(led|leads|leading|co-led|co-leads)\s+(the|this|its)?\s*(round|raise|financing|investment)", t)
                or re.search(r"\blead(ing)?\s+investors?\s*(was|were|is|are|:)?[^.\n]{0,60}" + n.pattern, t))


def normalize_round(r: str | None) -> str | None:
    if not r:
        return None
    t = re.sub(r"\s+", " ", str(r).strip().lower())
    if t in ("unknown", "series unknown", "n/a", "none", "undisclosed", "funding", "round"):
        return None
    m = re.search(r"series\s*([a-h])(\d?)", t)
    if m:
        return f"Series {m.group(1).upper()}{m.group(2)}"
    if "pre-seed" in t or "pre seed" in t:
        return "Pre-Seed"
    if "seed" in t:
        return "Seed"
    return str(r).strip()

# ------------------------------------------------------------------ 4. normalize investors
ALIASES = {
    "a16z": "andreessen horowitz", "a16z speedrun": "andreessen horowitz", "andreessen horowitz a16z": "andreessen horowitz",
    "yc": "y combinator", "ycombinator": "y combinator", "y combinator yc": "y combinator",
    "gc": "general catalyst", "kpcb": "kleiner perkins", "kleiner perkins caufield byers": "kleiner perkins",
    "sequoia capital": "sequoia", "sequoia capital china": "sequoia china", "lsvp": "lightspeed",
    "lightspeed venture partners": "lightspeed", "nea": "new enterprise associates", "gv": "google ventures",
    "founders fund": "founders fund", "khosla": "khosla", "sv angel": "sv angel", "svangel": "sv angel",
}
_INV_SUFFIX = re.compile(r"\b(capital|ventures?|venture partners|partners|management|vc|fund|funds|group|"
                         r"holdings|llc|lp|l\.p\.|inc|investments?|advisors)\b\.?", re.I)


def investor_key(name: str) -> str:
    """Canonical comparison key: lowercase, '&'->'and', aliases, legal/fund suffixes dropped."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = ALIASES.get(s, s)
    core = re.sub(r"\s+", " ", _INV_SUFFIX.sub(" ", s)).strip() or s
    return ALIASES.get(core, core)


def canonical_names(raw_names: list[str]) -> dict[str, str]:
    """key -> display name (the most frequent raw spelling)."""
    by_key: dict[str, Counter] = defaultdict(Counter)
    for n in raw_names:
        if n and n.strip():
            by_key[investor_key(n)][n.strip()] += 1
    return {k: c.most_common(1)[0][0] for k, c in by_key.items()}


# ------------------------------------------------------------------ pipeline
async def run_investor_collection(entities: list[dict], use_rss: bool = False, limit: int | None = None,
                                  only_ids: set[str] | None = None, model: str | None = None) -> dict:
    from .sources import crawl_pages

    ents = [e for e in entities if not only_ids or e["entity_id"] in only_ids][:limit]
    by_id = {e["entity_id"]: e for e in ents}
    cands = {k: v for k, v in harvest_yc_news(ents).items() if k in by_id}
    stats = Counter(companies=len(ents), with_yc_news=len(cands))
    if use_rss:
        for e in ents:
            if e["entity_id"] not in cands:
                try:
                    items = await rss_candidates(e["name"])
                except Exception:  # noqa: BLE001
                    items = []
                await asyncio.sleep(1.0)                      # be polite
                if items:
                    cands[e["entity_id"]] = items
                    stats["with_rss_news"] += 1

    jobs = [(cid, it) for cid, items in cands.items() for it in items]
    stats["articles"] = len(jobs)
    pages = {p["url"]: p["markdown"] for p in await crawl_pages([it["url"] for _, it in jobs], source="news")}
    stats["articles_fetched"] = len(pages)

    usage, records = Usage(), []
    for cid, it in jobs:
        ev = await asyncio.to_thread(extract_event, by_id[cid], it, pages.get(it["url"]), model, usage)
        if ev is None:
            stats["parse_fail"] += 1
            continue
        if not (ev.about_company and ev.is_funding):
            stats["rejected_not_about_company_or_not_funding"] += 1
            continue
        records.append({"source": "news", "kind": "funding",
                        "source_id": hashlib.sha1(f"{cid}|{it['url']}".encode()).hexdigest()[:16],
                        "name": by_id[cid]["name"],
                        "data": {"entity_id": cid, "url": it["url"], "title": it["title"], "via": it.get("via"),
                                 "news_date": it.get("date"), "body_available": it["url"] in pages,
                                 **ev.model_dump(exclude={"about_company", "is_funding"})}})
    stats["funding_events"] = len(records)
    stats["events_with_investors"] = sum(1 for r in records if r["data"]["investors"] or r["data"]["lead_investors"])
    mongo.save_records(records)
    stats["llm"] = usage.summary()
    return dict(stats)


def merge_into_entities(entities: list[dict], records: list[dict]) -> tuple[list[dict], dict]:
    """Attach normalized investors (with round/amount/date/lead/source) to each entity."""
    names = canonical_names([n for r in records for n in r["data"].get("investors", []) + r["data"].get("lead_investors", [])])
    by_ent: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        d = r["data"]
        leads = {investor_key(n) for n in d.get("lead_investors", [])}
        for raw in set(d.get("investors", [])) | set(d.get("lead_investors", [])):
            k = investor_key(raw)
            if not k or normalize_name(raw) == normalize_name(r["name"]):
                continue                                         # the company is not its own investor
            cur = by_ent[d["entity_id"]].setdefault(k, {"name": names.get(k, raw), "key": k, "rounds": [],
                                                       "lead": False, "source_urls": []})
            cur["rounds"].append({"round": d.get("round"), "amount": d.get("amount_usd"),
                                  "date": d.get("announced_date") or d.get("news_date")})
            cur["lead"] = cur["lead"] or k in leads
            cur["source_urls"].append(d["url"])
    
    for inv_map in by_ent.values():
        for v in inv_map.values():
            seen, uniq = set(), []
            for r in v["rounds"]:
                key = ("amt", r["amount"]) if r.get("amount") else ("round", r.get("round"))
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            v["rounds"] = uniq   

    n_edges = 0
    for e in entities:
        new = by_ent.get(e["entity_id"], {})
        keep = [i for i in e.get("investors", []) if investor_key(i["name"] if isinstance(i, dict) else i) not in new]
        inv = []
        for v in new.values():
            last = sorted(v["rounds"], key=lambda x: str(x.get("date") or ""))[-1]
            inv.append({"name": v["name"], "round": last.get("round"), "amount": last.get("amount"),
                        "date": last.get("date"), "lead": v["lead"], "n_rounds": len(v["rounds"]),
                        "source_url": v["source_urls"][0]})
        e["investors"] = keep + inv
        n_edges += len(inv)
    return entities, {"investors": len(names), "investor_edges": n_edges,
                      "companies_with_investors": sum(1 for v in by_ent.values() if v)}


def investor_node_id(name: str) -> str:
    return f"inv:{slugify(investor_key(name)) or slugify(name)}"