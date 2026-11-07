"""Build an evaluation set (question + reference answer + reference node ids) from REAL data.

Ground truth comes from facts the sources state directly, not from the walk or from
the derived SIMILAR_TO edges, so the evaluation is not circular:
  competitors   : same YC sub-industry AND >= 1 shared YC tag
  founders      : founder names on the YC company page
  market_signal : YC industry x batch membership
  funding       : SEC Form D amounts sold
  network       : people linked to >= 2 companies (founder / Form D executive)
  investors     : investors named on crawled pages (only if any were extracted)
Review the generated file by hand before reporting numbers: tags can be broad.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..collect.resolve import slugify
from ..collect.investors import investor_node_id
from ..kg.builder import person_id

# YC tags too broad to signal competition on their own
GENERIC_TAGS = {"ai", "artificial intelligence", "generative ai", "machine learning", "ai assistant", "b2b",
                "saas", "enterprise", "enterprise software", "consumer", "marketplace", "finance", "fintech",
                "analytics", "automation", "api", "developer tools", "open source", "hardware", "hard tech"}


def _tags(e: dict) -> set[str]:
    return {t.lower() for t in ((e.get("attrs") or {}).get("tags") or [])} - GENERIC_TAGS

def _one(e: dict) -> str:
    ol = (e.get("attrs") or {}).get("one_liner")
    return f"{e['name']} ({ol})" if ol else e["name"]


def make_questions(entities: list[dict], per_type: int = 6, seed: int = 0,
                                      max_refs: int = 8, min_shared_tags: int = 2, kinds: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    ents = [e for e in entities if e.get("type", "company") == "company"]
    qs: dict[str, list[dict]] = defaultdict(list)

    # --- competitors
    by_sub = defaultdict(list)
    for e in ents:
        a = e.get("attrs", {})
        key = a.get("subindustry") or a.get("industry")
        if key:
            by_sub[key].append(e)
    for sub, group in by_sub.items():
        for tgt in group:
            tags = _tags(tgt)
            comps = [c for c in group if c is not tgt and len(tags & _tags(c)) >= min_shared_tags]
            if 1 <= len(comps) <= max_refs:
                shared = sorted(tags & {t for c in comps for t in _tags(c)})
                qs["competitors"].append({
                    "id": f"comp-{slugify(tgt['name'])}", "type": "competitors",
                    "question": f"Which startups compete with {tgt['name']}?",
                    "reference": f"{_one(tgt)}. Startups in the same YC sub-industry ({sub}) with overlapping "
                                 f"focus ({', '.join(shared)}): " + "; ".join(_one(c) for c in comps) + ".",
                    "reference_ids": [c["entity_id"] for c in comps]})

    # --- founders
    for e in ents:
        fs = [f for f in e.get("founders", []) if f.get("name")]
        if fs:
            qs["founders"].append({
                "id": f"founder-{slugify(e['name'])}", "type": "founders",
                "question": f"Who founded {e['name']}?",
                "reference": f"{e['name']} was founded by " + ", ".join(
                    f"{f['name']}" + (f" ({f['title']})" if f.get("title") else "") for f in fs) + ".",
                "reference_ids": [person_id(f["name"], e["name"]) for f in fs]})

    # --- market signals: industry x batch
    by_ib = defaultdict(list)
    for e in ents:
        a = e.get("attrs", {})
        if a.get("industry") and a.get("batch"):
            by_ib[(a["industry"], a["batch"])].append(e)
    for (ind, batch), group in by_ib.items():
        if 2 <= len(group) <= max_refs + 4:
            qs["market_signal"].append({
                "id": f"emerging-{slugify(ind)}-{slugify(batch)}", "type": "market_signal",
                "question": f"What {ind} startups are in the YC {batch} batch, and what are they building?",
                "reference": f"YC {batch} {ind} startups: " + "; ".join(_one(c) for c in group) + ".",
                "reference_ids": [c["entity_id"] for c in group]})

    # --- funding from SEC Form D
    for e in ents:
        fil = [f for f in e.get("filings", []) if f.get("total_amount_sold")]
        if fil:
            total = sum(f["total_amount_sold"] for f in fil)
            qs["funding"].append({
                "id": f"funding-{slugify(e['name'])}", "type": "lookup",
                "question": f"How much has {e['name']} raised according to its SEC Form D filings?",
                "reference": f"{e['name']} reported ${total:,.0f} sold across {len(fil)} Form D filing(s): "
                             + "; ".join(f"{f['accession']} ${f['total_amount_sold']:,.0f}"
                                         f" (first sale {f.get('date_first_sale') or 'n/a'})" for f in fil) + ".",
                "reference_ids": [e["entity_id"]] + [f"filing:{f['accession']}" for f in fil]})

    # --- founder / executive networks
    person_cos = defaultdict(set)
    for e in ents:
        for f in e.get("founders", []):
            person_cos[f["name"]].add(e["entity_id"])
        for f in e.get("filings", []):
            for p in f.get("related_persons", []):
                person_cos[p["name"]].add(e["entity_id"])
    name_of = {e["entity_id"]: e["name"] for e in ents}
    for person, cos in person_cos.items():
        if len(person.split()) > 1 and 2 <= len(cos) <= max_refs:   # bare first names are ambiguous
            qs["network"].append({
                "id": f"network-{slugify(person)}", "type": "founders",
                "question": f"Which companies is {person} connected to?",
                "reference": f"{person} is linked to: " + ", ".join(sorted(name_of[c] for c in cos)) + ".",
                "reference_ids": sorted(cos)})

    # --- investors (only if extracted)
    for e in ents:
        invs = [i["name"] if isinstance(i, dict) else i for i in e.get("investors", [])]
        if invs:
            qs["investors"].append({
                "id": f"inv-{slugify(e['name'])}", "type": "investors",
                "question": f"Which investors have backed {e['name']}?",
                "reference": f"{e['name']} is backed by " + ", ".join(invs) + ".",
                "reference_ids": [investor_node_id(i) for i in invs]})

    # --- open-ended exploration: no entity is named; the answer is every company with a specific tag
    sector_names = {s.lower() for e in ents for s in e.get("sectors", [])}
    company_names = {e["name"].lower() for e in ents}
    by_tag = defaultdict(list)
    for e in ents:
        for t in (e.get("attrs") or {}).get("tags") or []:
            tl = t.lower()
            if tl not in GENERIC_TAGS and tl not in sector_names and tl not in company_names:
                by_tag[t].append(e)
    for tag, group in by_tag.items():
        if 3 <= len(group) <= max_refs + 4:
            qs["open"].append({
                "id": f"open-{slugify(tag)}", "type": "exploration",
                "question": f"Which YC startups are working on {tag}?",
                "reference": f"YC startups tagged {tag}: " + "; ".join(_one(c) for c in group) + ".",
                "reference_ids": [c["entity_id"] for c in group]})

    # --- multi-hop: company -> shared founder/executive -> other company (2 hops from the anchor)
    by_person = defaultdict(set)
    for e in ents:
        for f in e.get("founders", []):
            if f.get("name"):
                by_person[person_id(f["name"], e["name"])].add(e["entity_id"])
        for f in e.get("filings", []):
            for p in f.get("related_persons", []):
                by_person[person_id(p["name"], e["name"])].add(e["entity_id"])
    linked = defaultdict(set)
    for cos in by_person.values():
        if 2 <= len(cos) <= max_refs:
            for c in cos:
                linked[c] |= cos - {c}
    for cid, others in linked.items():
        qs["multihop"].append({
            "id": f"multihop-{slugify(name_of[cid])}", "type": "multihop",
            "question": f"Which other YC companies share a founder or executive with {name_of[cid]}?",
            "reference": f"{name_of[cid]} shares a founder or executive with: "
                         + ", ".join(sorted(name_of[c] for c in others)) + ".",
            "reference_ids": sorted(others)})

    # --- investor networks (2 hops). Y Combinator backs every company, so it is never an anchor or an answer.
    yc = investor_node_id("Y Combinator")
    portfolio, inv_name, inv_of_co = defaultdict(set), {}, defaultdict(set)
    for e in ents:
        for i in e.get("investors", []):
            n = i["name"] if isinstance(i, dict) else i
            nid = investor_node_id(n)
            if nid == yc:
                continue
            portfolio[nid].add(e["entity_id"])
            inv_name.setdefault(nid, n)
            inv_of_co[e["entity_id"]].add(nid)
    # investor -> company -> other investor
    for nid, cos in portfolio.items():
        partners = set().union(*(inv_of_co[c] for c in cos)) - {nid}
        if 2 <= len(partners) <= max_refs + 4:
            qs["coinvest"].append({
                "id": f"coinvest-{slugify(inv_name[nid])}", "type": "coinvest",
                "question": f"Which investors have co-invested with {inv_name[nid]} in YC startups?",
                "reference": f"{inv_name[nid]} invested in " + ", ".join(sorted(name_of[c] for c in cos))
                             + "; co-investors in those rounds: " + ", ".join(sorted(inv_name[p] for p in partners)) + ".",
                "reference_ids": sorted(partners)})
    # company -> investor -> other company
    for cid, invs in inv_of_co.items():
        others = set().union(*(portfolio[i] for i in invs)) - {cid}
        if 2 <= len(others) <= max_refs + 4:
            qs["shared_investor"].append({
                "id": f"sharedinv-{slugify(name_of[cid])}", "type": "shared_investor",
                "question": f"Which other YC companies share an investor with {name_of[cid]}?",
                "reference": f"{name_of[cid]} is backed by " + ", ".join(sorted(inv_name[i] for i in invs))
                             + "; they also backed: " + ", ".join(sorted(name_of[c] for c in others)) + ".",
                "reference_ids": sorted(others)})
            
    out = []
    for t, lst in qs.items():
        if kinds and t not in kinds:
            continue
        rng.shuffle(lst)
        out += lst[:per_type]
    return out


def write_questions(entities: list[dict], path: str | Path, **kw) -> dict:
    qs = make_questions(entities, **kw)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    counts = defaultdict(int)
    for q in qs:
        counts[q["id"].split("-")[0]] += 1
    return {"path": str(p), "n": len(qs), "by_kind": dict(counts)}
