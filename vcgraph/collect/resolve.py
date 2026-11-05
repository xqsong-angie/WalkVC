"""Entity resolution: merge duplicate startup records across YC / SEC / OpenCorporates.

Pipeline
  1. normalise names (strip legal suffixes, punctuation, 'the') + extract web domain
  2. block on first normalised token to avoid O(n^2)
  3. pairwise rules:  same domain -> match ; fuzzy >= HIGH -> match ;
                      LOW <= fuzzy < HIGH and same company_hint/state -> match
                      (optional LLM adjudication for the grey zone)
  4. union-find -> clusters -> canonical entity with merged attributes + provenance
"""
from __future__ import annotations

import re
import unicodedata
import urllib.parse
from collections import Counter, defaultdict

from rapidfuzz import fuzz

from ..jsonparse import parse_json
from ..llm import chat

LEGAL_SUFFIXES = r"(incorporated|inc|corp|corporation|co|company|llc|l\.l\.c|ltd|limited|plc|gmbh|sa|sas|bv|pbc|lp|llp|holdings?|technologies)$"
HIGH, LOW = 93, 85


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s)                     # "(CIK 000123)" etc.
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"^the\s+", "", s.strip())
    for _ in range(2):
        s = re.sub(r"\s+" + LEGAL_SUFFIXES, "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def domain_of(url: str | None) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host or None


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize_name(s) or s.lower()).strip("-") or "unknown"


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _llm_same(a: dict, b: dict, model: str | None) -> bool:
    prompt = ("Do these two records refer to the same company? Answer JSON {\"same\": true|false}.\n"
              f"A: {a['name']} | {a['data'].get('website') or a['data'].get('state') or ''}\n"
              f"B: {b['name']} | {b['data'].get('website') or b['data'].get('state') or ''}")
    text, _ = chat([{"role": "user", "content": prompt}], model, json_mode=True, max_tokens=20, purpose="er")
    obj = parse_json(text, kind="er")
    return bool(obj and obj.get("same"))


def resolve(records: list[dict], use_llm: bool = False, model: str | None = None) -> tuple[list[dict], dict]:
    """Return (canonical company entities, stats)."""
    n = len(records)
    keys = [normalize_name(r.get("data", {}).get("company_hint") or r["name"]) if r["kind"] != "company"
            else normalize_name(r["name"]) for r in records]
    names = [normalize_name(r["name"]) for r in records]
    doms = [domain_of(r.get("data", {}).get("website")) for r in records]
    uf = _UF(n)


    # Each YC record is already a unique company (unique slug): never merge two of them, and
    # never attach a SEC / OpenCorporates record when several YC companies share that name.
    is_co = [r["kind"] == "company" for r in records]
    has_co = list(is_co)                      # per union-find root: cluster already holds a company
    name_count = Counter(names[i] for i in range(n) if is_co[i])

    def link(a: int, b: int) -> int:
        ra_, rb_ = uf.find(a), uf.find(b)
        if ra_ == rb_ or (has_co[ra_] and has_co[rb_]):
            return 0
        co = a if is_co[a] else b if is_co[b] else None
        if co is not None and name_count[names[co]] > 1:
            return 0                          # ambiguous: two YC companies with this name
        uf.p[ra_] = rb_
        has_co[rb_] = has_co[rb_] or has_co[ra_]
        return 1
    
    blocks: dict[str, list[int]] = defaultdict(list)
    for i in range(n):
        for k in {keys[i], names[i]}:
            if k:
                blocks[k.split(" ")[0]].append(i)

    pairs_checked = merges = llm_calls = 0
    for idxs in blocks.values():
        idxs = sorted(set(idxs))
        for a_i, a in enumerate(idxs):
            for b in idxs[a_i + 1:]:
                if uf.find(a) == uf.find(b):
                    continue
                pairs_checked += 1
                ra, rb = records[a], records[b]
                if doms[a] and doms[a] == doms[b]:
                    merges += link(a, b); continue
                score = max(fuzz.token_sort_ratio(names[a], names[b]),
                            fuzz.token_sort_ratio(keys[a], keys[b]))
                if score >= HIGH:
                    merges += link(a, b)
                elif score >= LOW:
                    hint_match = (keys[a] == keys[b]) or (
                        ra["data"].get("state") and ra["data"].get("state") == rb["data"].get("state"))
                    same = bool(hint_match)
                    if not same and use_llm:
                        llm_calls += 1
                        same = _llm_same(ra, rb, model)
                    if same:
                        merges += link(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    entities = [_merge([records[i] for i in idx]) for idx in clusters.values()]
    entities = [e for e in entities if e is not None]
    stats = {"records": n, "entities": len(entities), "pairs_checked": pairs_checked,
             "merges": merges, "llm_calls": llm_calls,
             "multi_source_entities": sum(1 for e in entities if len(e["sources"]) > 1)}
    return entities, stats


_PRIORITY = {"yc": 0, "demo": 0, "opencorporates": 1, "sec": 2}


def _merge(recs: list[dict]) -> dict | None:
    recs = sorted(recs, key=lambda r: _PRIORITY.get(r["source"], 9))
    companies = [r for r in recs if r["kind"] == "company"]
    filings = [r for r in recs if r["kind"] == "filing"]
    legal = [r for r in recs if r["kind"] == "legal_entity"]
    base = companies[0] if companies else recs[0]
    name = base["name"] if companies else (base["data"].get("company_hint") or base["name"])
    if not name:
        return None
    attrs: dict = {}
    for r in companies:
        for k, v in r["data"].items():
            if v not in (None, "", []) and k not in attrs:
                attrs[k] = v
    founders = {f["name"]: f for r in companies for f in r["data"].get("founders", []) if f.get("name")}
    investors = sorted({i for r in companies for i in r["data"].get("investors", []) if i})
    sectors = sorted({s for r in companies for s in r["data"].get("sectors", []) if s}
                     | {f["data"].get("industry_group") for f in filings if f["data"].get("industry_group")})
    return {
        "entity_id": "co:" + (re.sub(r"[^a-z0-9-]+", "-", companies[0]["source_id"].lower()).strip("-")
                              if companies else "unmatched-" + slugify(name)),
        "type": "company",
        "name": name,
        "aliases": sorted({r["name"] for r in recs if r["name"] and r["name"] != name}),
        "sources": sorted({r["source"] for r in recs}),
        "source_ids": [f"{r['source']}:{r['source_id']}" for r in recs],
        "attrs": {k: v for k, v in attrs.items() if k not in ("founders", "investors", "sectors")},
        "sectors": sectors,
        "founders": list(founders.values()),
        "investors": investors,
        "filings": [f["data"] | {"accession": f["source_id"]} for f in filings],
        "legal_entities": [l["data"] | {"id": l["source_id"], "name": l["name"]} for l in legal],
    }
