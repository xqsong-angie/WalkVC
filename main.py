"""VCGraph CLI.

  python main.py demo   [--backend neo4j|memory]          # synthetic graph + questions, no crawling
  python main.py collect --batches W25 S25 --limit 40 --yc-mode hybrid   # real YC data
  python main.py build                                       # rebuild Neo4j from Mongo `entities`
  python main.py questions --out data/yc/questions.jsonl     # eval set from real data
  python main.py ask "Which startups compete with Nimbusly?" [--model qwen2.5:14b] [--mode walk|cypher]
  python main.py eval --questions data/demo/questions.jsonl --models llama3.1:8b qwen2.5:14b gpt-4o
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

DEMO_DIR = Path("data/demo")
MEM_GRAPH = DEMO_DIR / "graph_memory.json"


def _graph(backend: str):
    from vcgraph.storage.graph import InMemoryGraph, Neo4jGraph

    if backend == "neo4j":
        return Neo4jGraph()
    if not MEM_GRAPH.exists():
        sys.exit("No in-memory graph yet: run `python main.py demo --backend memory` first.")
    return InMemoryGraph.load(MEM_GRAPH)


def cmd_demo(a):
    from vcgraph.kg.builder import build_graph
    from vcgraph.kg.demo import write_demo
    from vcgraph.storage.graph import InMemoryGraph, Neo4jGraph

    ents_path, q_path = write_demo(DEMO_DIR, n_companies=a.n)
    ents = json.loads(ents_path.read_text())
    print(f"generated {len(ents)} synthetic companies -> {ents_path}; questions -> {q_path}")
    g = Neo4jGraph() if a.backend == "neo4j" else InMemoryGraph()
    stats = build_graph(ents, g)
    if a.backend == "memory":
        g.save(MEM_GRAPH)
        print(f"saved in-memory graph -> {MEM_GRAPH}")
    print(json.dumps(stats, indent=1))


def cmd_collect(a):
    from vcgraph.collect.agent import run_collection

    stats = asyncio.run(run_collection(a.batches, limit=a.limit or None, yc_mode=a.yc_mode, skip_sec=a.skip_sec,
                                       skip_kg=a.skip_kg, use_llm_er=a.llm_er))
    print(json.dumps(stats, indent=1, default=str))


def cmd_build(a):
    from vcgraph.kg.builder import build_graph
    from vcgraph.storage import mongo
    from vcgraph.storage.graph import Neo4jGraph

    ents = mongo.load_entities()
    print(f"{len(ents)} entities from Mongo")
    print(json.dumps(build_graph(ents, Neo4jGraph()), indent=1))


def cmd_investors(a):
    """Funding news -> investor edges (YC 'Latest News' links, optional Google News RSS)."""
    import csv

    from vcgraph.collect.investors import merge_into_entities, run_investor_collection
    from vcgraph.collect.resolve import normalize_name
    from vcgraph.kg.builder import build_graph
    from vcgraph.storage import mongo
    from vcgraph.storage.graph import Neo4jGraph

    ents = mongo.load_entities()
    only = None
    if a.names_file:                       # e.g. the PitchBook gold companies, to evaluate early
        with open(a.names_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        col = next(c for c in ("Company Name", "Companies", "name") if rows and c in rows[0])
        wanted = {normalize_name(r[col]) for r in rows if r.get(col)}
        only = {e["entity_id"] for e in ents if normalize_name(e["name"]) in wanted}
        print(f"{len(only)} of {len(wanted)} names matched to companies in the graph")
    if not a.skip_collect:
        stats = asyncio.run(run_investor_collection(ents, use_rss=a.rss, limit=a.limit, only_ids=only,
                                                    model=a.model))
        print(json.dumps(stats, indent=1, default=str))
    ents, st = merge_into_entities(ents, mongo.load_records("news"))
    mongo.save_entities(ents)
    print(json.dumps(st, indent=1))
    if not a.no_build:
        g = Neo4jGraph()
        print(json.dumps(build_graph(ents, g), indent=1))
        g.close()


def cmd_questions(a):
    from vcgraph.eval.make_questions import write_questions
    from vcgraph.storage import mongo

    ents = mongo.load_entities()
    if not ents:
        sys.exit("No entities in Mongo yet: run `python main.py collect ...` first.")
    print(json.dumps(write_questions(ents, a.out, per_type=a.per_type, seed=a.seed,
                                     kinds=set(a.kinds) if a.kinds else None), indent=1))


def cmd_export_list(a):
    """CSV of YC companies (name + website) for PitchBook 'Upload a List'."""
    import csv

    from vcgraph.collect.resolve import domain_of

    if a.from_mongo:
        from vcgraph.storage import mongo
        rows = [(e["name"], (e.get("attrs") or {}).get("website") or "", (e.get("attrs") or {}).get("batch") or "")
                for e in mongo.load_entities()]
    else:  # straight from the YC JSON mirror, no Mongo / Docker needed
        import httpx

        from vcgraph.collect.sources import YC_OSS, normalize_batch
        rows = []
        for b in a.batches:
            slug = normalize_batch(b).lower().replace(" ", "-")
            r = httpx.get(f"{YC_OSS}/batches/{slug}.json", timeout=60)
            r.raise_for_status()
            rows += [(c.get("name", ""), c.get("website") or "", normalize_batch(c.get("batch") or b))
                     for c in r.json()]
    seen, out = set(), []
    for name, web, batch in rows:
        key = domain_of(web) or name.lower()
        if name and key not in seen:
            seen.add(key)
            out.append((name, web, batch))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Company Name", "Website", "YC Batch"])
        w.writerows(out)
    print(f"wrote {len(out)} companies -> {a.out}")


def cmd_ask(a):
    from vcgraph.reason.agent import ask

    g = _graph(a.backend)
    print("\n=== answer (streaming) ===")
    r = ask(a.question, g, model=a.model, mode=a.mode, check=not a.no_check,
            on_token=lambda t: print(t, end="", flush=True))
    print("\n")
    m = r["metrics"]
    print(f"intent={r['intent']} sub_queries={r['sub_queries']}")
    print(f"e2e={m['e2e_latency_s']}s ttft={m['ttft_s']}s timings={m['timings']}")
    print(f"tokens={m['llm']['total_tokens']} context_tokens={m['context_tokens']} db_roundtrips={m['db_roundtrips']}")
    for w in m["walks"]:
        print(f"  walk '{w['sub_query'][:60]}': iters={w['iterations']} stop={w['stop']} gains={w['gains']}")
    if m.get("cypher"):
        print(f"  cypher: {m['cypher']}")
    if a.verbose:
        print("\n=== evidence ===\n" + "\n".join(r["contexts"]))


def cmd_eval(a):
    from vcgraph.eval.run_eval import run_benchmark

    run_benchmark(_graph(a.backend), a.questions, a.models, a.modes, run_name=a.run_name,
                  ragas=not a.no_ragas, check=not a.no_check, limit=a.limit)


def main():
    p = argparse.ArgumentParser(description="Agentic KG exploration via random walks for VC intelligence")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo"); d.add_argument("--backend", default="neo4j", choices=["neo4j", "memory"])
    d.add_argument("--n", type=int, default=72); d.set_defaults(fn=cmd_demo)

    c = sub.add_parser("collect")
    c.add_argument("--batches", nargs="+", default=["W25"], help='e.g. W25 S25 or "Winter 2025"')
    c.add_argument("--limit", type=int, default=20, help="companies per batch (0 = all)")
    c.add_argument("--yc-mode", default="hybrid", choices=["hybrid", "json", "crawl"],
                   help="hybrid: JSON list + crawl YC pages for founders; json: no browser; "
                        "crawl: everything from www.ycombinator.com")
    c.add_argument("--skip-sec", action="store_true"); c.add_argument("--skip-kg", action="store_true")
    c.add_argument("--llm-er", action="store_true", help="LLM adjudication for grey-zone entity matches")
    c.set_defaults(fn=cmd_collect)

    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)

    el = sub.add_parser("export-list", help="CSV for PitchBook 'Upload a List'")
    el.add_argument("--batches", nargs="+", default=["W25"])
    el.add_argument("--from-mongo", action="store_true", help="use already-collected entities instead")
    el.add_argument("--out", default="data/pitchbook/yc_upload_list.csv")
    el.set_defaults(fn=cmd_export_list)
    
    iv = sub.add_parser("investors", help="extract investors from funding news and add them to the KG")
    iv.add_argument("--rss", action="store_true", help="Google News RSS for companies w/o funding news on YC")
    iv.add_argument("--limit", type=int, default=None)
    iv.add_argument("--names-file", default=None, help="CSV with a 'Company Name' / 'Companies' column")
    iv.add_argument("--model", default=None, help="extraction model (default EXTRACT_MODEL)")
    iv.add_argument("--skip-collect", action="store_true", help="only re-merge stored news records")
    iv.add_argument("--no-build", action="store_true", help="don't rebuild the Neo4j graph")
    iv.set_defaults(fn=cmd_investors)
    
    qq = sub.add_parser("questions")
    qq.add_argument("--out", default="data/yc/questions.jsonl")
    qq.add_argument("--per-type", type=int, default=6); qq.add_argument("--seed", type=int, default=0)
    qq.add_argument("--kinds", nargs="+", default=None,
                    help="subset: competitors founders market_signal funding network investors open multihop")
    qq.set_defaults(fn=cmd_questions)

    q = sub.add_parser("ask"); q.add_argument("question")
    q.add_argument("--model", default=None); q.add_argument("--mode", default="walk", choices=["walk", "cypher", "expand"])
    q.add_argument("--backend", default="neo4j", choices=["neo4j", "memory"])
    q.add_argument("--no-check", action="store_true"); q.add_argument("-v", "--verbose", action="store_true")
    q.set_defaults(fn=cmd_ask)

    e = sub.add_parser("eval")
    e.add_argument("--questions", default=str(DEMO_DIR / "questions.jsonl"))
    e.add_argument("--models", nargs="+", default=["llama3.1:8b", "qwen2.5:14b", "gpt-4o"])
    e.add_argument("--modes", nargs="+", default=["walk", "cypher"])
    e.add_argument("--backend", default="neo4j", choices=["neo4j", "memory"])
    e.add_argument("--run-name", default=None); e.add_argument("--limit", type=int, default=None)
    e.add_argument("--no-ragas", action="store_true"); e.add_argument("--no-check", action="store_true")
    e.set_defaults(fn=cmd_eval)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
