"""Answer-level recall: a metric every retrieval method can share, including the Cypher baseline.

Retrieval recall needs node ids, which Cypher results do not have. Answer recall instead checks the
final answer text: a gold entity counts as found if the answer cites its node id or mentions its name.

1) Runs the Cypher baseline on the two-hop / exploration sets if it has not been run yet.
2) Scores answer recall for every method on every set and prints one comparison table.

    python scripts/answer_recall.py              # needs Neo4j (to map gold ids -> names) and Ollama
    python scripts/answer_recall.py --no-run     # only score existing results
Outputs results/answer_recall/{per_question.csv, summary.csv}.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from vcgraph.collect.investors import ALIASES, investor_key  # noqa: E402
from vcgraph.collect.resolve import normalize_name  # noqa: E402

MODELS = ["llama3.1:8b", "qwen2.5:14b"]
QSETS = {"point": "data/yc/questions.jsonl", "explore": "data/yc/questions_explore.jsonl",
         "coinv": "data/yc/questions_coinvest.jsonl"}
# (question set, run name, mode in that run) -> method label
RUNS = [
    ("point", "yc_v1", "cypher", "cypher"),
    ("point", "yc_v2_anchor_only", "walk", "anchor_only"),
    ("point", "yc_v2", "walk", "anchor+walk_12"),
    ("explore", "explore_cypher", "cypher", "cypher"),
    ("explore", "explore_anchor_only", "walk", "anchor_only"),
    ("explore", "explore_v1", "walk", "anchor+walk_12"),
    ("explore", "explore_walk_40", "walk", "anchor+walk_40"),
    ("explore", "explore_expand_7000", "expand", "expand_7k"),
    ("explore", "explore_v1", "expand", "expand_16k"),
    ("coinv", "coinv_cypher", "cypher", "cypher"),
    ("coinv", "coinv_anchor_only", "walk", "anchor_only"),
    ("coinv", "coinv_v1", "walk", "anchor+walk_12"),
    ("coinv", "coinv_walk_40", "walk", "anchor+walk_40"),
    ("coinv", "coinv_expand_7000", "expand", "expand_7k"),
    ("coinv", "coinv_v1", "expand", "expand_16k"),
]
ORDER = ["cypher", "anchor_only", "anchor+walk_12", "anchor+walk_40", "expand_7k", "expand_16k"]


# reruns on the final graph (scripts/final_rerun.py); coinv_* and explore_cypher already ran on it
FINAL_RENAME = {"yc_v1": "point_final_cypher", "yc_v2_anchor_only": "point_final_anchor_only",
                "yc_v2": "point_final_walk12", "explore_anchor_only": "explore_final_anchor_only",
                "explore_v1": "explore_final_v1", "explore_walk_40": "explore_final_walk_40",
                "explore_expand_7000": "explore_final_expand_7000"}

def run_missing_cypher():
    for qset, run, mode, _ in RUNS:
        if mode == "cypher" and not (ROOT / "results" / run / "raw.jsonl").exists():
            cmd = [sys.executable, "main.py", "eval", "--questions", QSETS[qset], "--models", *MODELS,
                   "--modes", "cypher", "--no-ragas", "--run-name", run]
            print(f"\n>>> {run}")
            subprocess.run(cmd, check=True)


def _norm(s: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def variants(gid: str, name: str) -> set[str]:
    v = {name, normalize_name(name)}
    if gid.startswith("inv:"):
        k = investor_key(name)
        v.add(k)
        v.update(a for a, target in ALIASES.items() if target == k)      # "a16z" for Andreessen Horowitz
    return {_norm(x) for x in v if x and len(x.strip()) >= 3}


def answer_hits(answer: str, gold: dict[str, str]) -> float | None:
    if not gold:
        return None
    a = _norm(answer)
    cited = set(re.findall(r"\[([a-z]+:[^\]\s]+)\]", answer))
    hit = sum(1 for gid, name in gold.items() if gid in cited or any(v in a for v in variants(gid, name)))
    return hit / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--final", action="store_true", help="use the final-graph reruns (scripts/final_rerun.py)")
    args = ap.parse_args()
    global RUNS
    if args.final:
        RUNS = [(qs, FINAL_RENAME.get(run, run), mode, label) for qs, run, mode, label in RUNS]
    if not args.no_run:
        run_missing_cypher()
    from vcgraph.storage.graph import Neo4jGraph

    g = Neo4jGraph()
    questions = {}
    for qset, path in QSETS.items():
        for line in Path(path).read_text().splitlines():
            q = json.loads(line)
            questions[(qset, q["id"])] = q
    all_ids = sorted({i for q in questions.values() for i in q["reference_ids"]})
    names = {i: n.get("name", i) for i, n in g.get_nodes(all_ids).items()}
    g.close()

    rows = []
    for qset, run, mode, label in RUNS:
        path = ROOT / "results" / run / "raw.jsonl"
        if not path.exists():
            print(f"(missing {run})")
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            q = questions.get((qset, r["qid"]))
            if r["mode"] != mode or q is None:
                continue
            gold = {i: names.get(i, i) for i in q["reference_ids"]}
            rows.append({"set": qset, "method": label, "model": r["model"], "qid": r["qid"],
                         "kind": r["qid"].split("-")[0], "answer_recall": answer_hits(r.get("answer", ""), gold),
                         "ctx_tokens": r["metrics"].get("context_tokens")})
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], ORDER, ordered=True)
    out = ROOT / "results" / ("answer_recall_final" if args.final else "answer_recall")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_question.csv", index=False)
    summary = df.pivot_table(index=["set", "model"], columns="method", values="answer_recall",
                             aggfunc="mean", observed=True).round(3)
    pd.set_option("display.width", 200)
    print("\n=== answer recall (share of gold entities named or cited in the final answer) ===")
    print(summary.to_string())
    print("\n=== by kind ===")
    print(df.pivot_table(index=["kind", "model"], columns="method", values="answer_recall",
                         aggfunc="mean", observed=True).round(3).to_string())
    summary.to_csv(out / "summary.csv")
    print(f"\nsaved -> {out}/per_question.csv, summary.csv")


if __name__ == "__main__":
    main()