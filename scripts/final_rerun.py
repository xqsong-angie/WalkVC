"""Re-run the main configurations on ONE final graph, so every number in the report comes from it.

The point-query runs (yc_v1 / yc_v2*) and the explore runs were made before the investor edges were
added; the coinvest runs, the stopping ablation and the explore Cypher run were made after. This
script (1) removes Investor nodes left over from the old id scheme, then (2) re-runs the old ones:

  point set   : point_final_cypher, point_final_anchor_only, point_final_walk12
  explore set : explore_final_{anchor_only, v1, expand_2500, expand_7000, walk_40} (via explore_budget.py)

    caffeinate -i python scripts/final_rerun.py        # ~4-5 hours, skips runs that already exist
Afterwards:
    python scripts/answer_recall.py --final --no-run
    python scripts/two_hop_summary.py --explore-tag explore_final
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
MODELS = ["llama3.1:8b", "qwen2.5:14b"]
POINT = "data/yc/questions.jsonl"
POINT_RUNS = {  # run name -> (modes, env)
    "point_final_cypher": (["cypher"], {}),
    "point_final_anchor_only": (["walk"], {"WALK_MAX_ITERS": "0", "WALK_TOP_NODES": "12"}),
    "point_final_walk12": (["walk"], {"WALK_TOP_NODES": "12"}),
}


def remove_stale_investors():
    """Investor nodes whose id is no longer produced from the current entities (old id scheme)."""
    from vcgraph.collect.investors import investor_node_id
    from vcgraph.storage import mongo
    from vcgraph.storage.graph import Neo4jGraph

    keep = {investor_node_id(i["name"] if isinstance(i, dict) else i)
            for e in mongo.load_entities() for i in e.get("investors", [])}
    g = Neo4jGraph()
    before = g._run("MATCH (n:Investor) RETURN count(n) AS c")[0]["c"]
    g._run("MATCH (n:Investor) WHERE NOT n.id IN $keep DETACH DELETE n", keep=sorted(keep))
    after = g._run("MATCH (n:Investor) RETURN count(n) AS c")[0]["c"]
    g.finalize()                                             # recompute degrees
    g.close()
    print(f"Investor nodes: {before} -> {after} ({before - after} stale removed)")


def main():
    remove_stale_investors()
    for run, (modes, env) in POINT_RUNS.items():
        if (ROOT / "results" / run / "raw.jsonl").exists():
            print(f"(exists) {run}")
            continue
        print(f"\n>>> {run} env={env}")
        subprocess.run([sys.executable, "main.py", "eval", "--questions", POINT, "--models", *MODELS,
                        "--modes", *modes, "--no-ragas", "--run-name", run],
                       env={**os.environ, **env}, check=True)
    subprocess.run([sys.executable, "scripts/explore_budget.py", "--questions", "data/yc/questions_explore.jsonl",
                    "--tag", "explore_final"], check=True)
    print("\nnext:\n  python scripts/answer_recall.py --final --no-run\n"
          "  python scripts/two_hop_summary.py --explore-tag explore_final")


if __name__ == "__main__":
    main()