"""Synthetic, clearly-fictional VC ecosystem for offline development and evaluation.

It plants known structure (competitor clusters, co-investor syndicates, founder alumni
networks, a recent 'emerging' batch) so that every generated question has a ground-truth
reference answer for RAGAS.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..collect.resolve import slugify

SECTORS = {
    "AI Infrastructure": ["GPU cluster scheduling for model training", "low-latency LLM inference serving",
                          "vector database for retrieval augmented generation", "evaluation platform for LLM agents"],
    "Climate Tech": ["carbon accounting software for manufacturers", "grid-scale battery analytics",
                     "methane leak detection with satellites", "EV fleet charging optimization"],
    "Fintech": ["embedded payments API for vertical SaaS", "AI underwriting for small business loans",
                "treasury management for startups", "fraud detection for real-time payments"],
    "Healthcare AI": ["ambient clinical note generation", "prior authorization automation for clinics",
                      "radiology triage with computer vision", "remote patient monitoring for cardiology"],
    "Developer Tools": ["AI code review bot", "observability for serverless apps",
                        "feature flag platform", "end-to-end test generation with LLMs"],
    "Robotics": ["warehouse picking robots", "autonomous inspection drones",
                 "robotic kitchen automation", "humanoid robot teleoperation data"],
}
PREFIX = ["Nimbus", "Quark", "Lumen", "Vela", "Orbit", "Cobalt", "Fern", "Atlas", "Helix", "Kite", "Prism",
          "Tidal", "Ember", "Nova", "Pylon", "Sable", "Juno", "Rivet", "Halo", "Zephyr", "Onyx", "Mosaic",
          "Lattice", "Beacon", "Cinder", "Drift", "Echo", "Flint", "Grove", "Ion"]
SUFFIX = ["ly", "io", "Works", "Labs", "AI", "Stack", "Grid", "Mind", "Path", "Forge"]
FIRST = ["Ava", "Ben", "Chloe", "Dev", "Elena", "Farah", "Gabe", "Hana", "Ivan", "Jia", "Kofi", "Lena", "Mateo",
         "Nadia", "Omar", "Priya", "Quinn", "Ravi", "Sofia", "Theo", "Uma", "Victor", "Wen", "Ximena", "Yusuf", "Zoe"]
LAST = ["Arden", "Brook", "Calder", "Dane", "Ellison", "Frost", "Garnet", "Hale", "Iver", "Jett", "Kerr", "Lowe",
        "Marsh", "Noor", "Oakes", "Pryce", "Quill", "Rhee", "Stroud", "Tamsin", "Ueda", "Vance", "Wilde", "Yarrow"]
FUNDS = ["Northpeak Ventures", "Bluefjord Capital", "Kestrel Partners", "Saltmarsh Fund", "Ironleaf VC",
         "Copperline Ventures", "Meridian Seed", "Glasswing Angels", "Redwood Arc", "Harbor Light Capital",
         "Polaris Fieldworks", "Tern Street Ventures"]
BATCHES = ["W24", "S24", "W25", "S25"]


def generate(n_companies: int = 72, seed: int = 7) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    names = rng.sample([p + s for p in PREFIX for s in SUFFIX], n_companies)
    sectors = list(SECTORS)
    # investor focus: each fund mainly backs 2 sectors; syndicates co-invest
    focus = {f: rng.sample(sectors, 2) for f in FUNDS}
    entities, by_sector, by_product = [], defaultdict(list), defaultdict(list)
    used_people: set[str] = set()

    def person():
        while True:
            n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if n not in used_people:
                used_people.add(n)
                return n

    for i, name in enumerate(names):
        sector = sectors[i % len(sectors)]
        product = rng.choice(SECTORS[sector])
        batch = BATCHES[min(3, i * 4 // n_companies)]
        eid = f"co:{slugify(name)}"
        backers = [f for f in FUNDS if sector in focus[f] and rng.random() < 0.45][:3]
        amount = rng.choice([1.5, 2, 3, 4, 6, 10, 15]) * 1e6
        ent = {
            "entity_id": eid, "type": "company", "name": name, "aliases": [], "sources": ["demo"],
            "source_ids": [f"demo:{i}"], "sectors": [sector],
            "attrs": {"one_liner": f"{name} builds {product}.",
                      "description": f"{name} is a {sector} startup that builds {product}. "
                                     f"Founded in {2019 + i * 6 // n_companies}, based in "
                                     f"{rng.choice(['San Francisco', 'New York', 'Austin', 'Boston', 'London', 'Toronto'])}.",
                      "batch": batch, "team_size": rng.randint(3, 120), "status": "Active",
                      "founded_year": 2019 + i * 6 // n_companies, "website": f"https://{slugify(name)}.example"},
            "founders": [{"name": person(), "title": t, "previously": []} for t in ("CEO", "CTO")],
            "investors": [{"name": b, "round": "Seed" if batch in ("W25", "S25") else "Series A",
                           "amount": amount / max(1, len(backers))} for b in backers],
            "filings": [], "legal_entities": [],
            "_product": product,
        }
        if rng.random() < 0.6:
            ent["filings"].append({"accession": f"0009{i:06d}-25-{i:06d}", "industry_group": sector,
                                   "total_amount_sold": amount, "total_offering_amount": amount * 1.2,
                                   "date_first_sale": f"{2023 + i * 3 // n_companies}-{rng.randint(1, 12):02d}-15",
                                   "related_persons": [{"name": ent["founders"][0]["name"],
                                                        "roles": ["Executive Officer", "Director"]}]})
        entities.append(ent)
        by_sector[sector].append(ent)
        by_product[product].append(ent)

    # founder alumni network: ~35% of later founders previously worked at an earlier company
    for i, ent in enumerate(entities[n_companies // 3:], start=n_companies // 3):
        if rng.random() < 0.35:
            prev = rng.choice(entities[: i])
            ent["founders"][rng.randint(0, 1)]["previously"].append(
                {"company_id": prev["entity_id"], "company_name": prev["name"], "role": "Engineer"})

    questions = _questions(entities, by_sector, by_product, rng)
    for e in entities:
        e.pop("_product", None)
    return entities, questions


def _questions(entities, by_sector, by_product, rng) -> list[dict]:
    qs = []
    # 1) competitor discovery (same product line)
    for product, group in by_product.items():
        if len(group) >= 2 and len(qs) < 6:
            tgt = group[0]
            comps = [c["name"] for c in group[1:]]
            qs.append({"id": f"comp-{slugify(tgt['name'])}", "type": "competitors",
                       "question": f"Which startups compete with {tgt['name']}?",
                       "reference": f"{tgt['name']} builds {product}. Competitors building the same thing: "
                                    + ", ".join(comps) + ".",
                       "reference_ids": [c["entity_id"] for c in group[1:]]})
    # 2) investor graph by sector
    for sector, group in list(by_sector.items())[:4]:
        invs = sorted({i["name"] for c in group for i in c["investors"]})
        if not invs:
            continue
        qs.append({"id": f"inv-{slugify(sector)}", "type": "investors",
                   "question": f"Which investors are most active in {sector} startups?",
                   "reference": f"Investors that backed {sector} companies: " + ", ".join(invs) + ".",
                   "reference_ids": [f"inv:{slugify(i)}" for i in invs]})
    # 3) co-investors
    for fund in FUNDS[:3]:
        co = sorted({i["name"] for c in entities if fund in [x["name"] for x in c["investors"]]
                     for i in c["investors"] if i["name"] != fund})
        if co:
            qs.append({"id": f"coinv-{slugify(fund)}", "type": "investors",
                       "question": f"Who has co-invested alongside {fund}?",
                       "reference": f"Co-investors of {fund}: " + ", ".join(co) + ".",
                       "reference_ids": [f"inv:{slugify(i)}" for i in co]})
    # 4) founder networks
    alumni = [(c, f) for c in entities for f in c["founders"] if f["previously"]]
    for c, f in alumni[:4]:
        prev = f["previously"][0]
        qs.append({"id": f"founder-{slugify(f['name'])}", "type": "founders",
                   "question": f"Who founded {c['name']} and what is their background?",
                   "reference": f"{c['name']} was founded by " + " and ".join(x["name"] for x in c["founders"])
                                + f". {f['name']} previously worked at {prev['company_name']}.",
                   "reference_ids": [f"person:{slugify(x['name'])}" for x in c["founders"]] + [prev["company_id"]]})
    # 5) emerging market signals
    for sector, group in list(by_sector.items())[:3]:
        recent = [c for c in group if c["attrs"]["batch"] in ("W25", "S25")]
        if recent:
            qs.append({"id": f"emerging-{slugify(sector)}", "type": "market_signal",
                       "question": f"What are the emerging early-stage startups in {sector} from the most recent batches?",
                       "reference": f"Recent ({'/'.join(['W25', 'S25'])}) {sector} startups: "
                                    + "; ".join(f"{c['name']} ({c['attrs']['one_liner']})" for c in recent),
                       "reference_ids": [c["entity_id"] for c in recent]})
    return qs


def write_demo(out_dir: str | Path = "data/demo", **kw) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ents, qs = generate(**kw)
    (out / "entities.json").write_text(json.dumps(ents, indent=1))
    qpath = out / "questions.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    return out / "entities.json", qpath
