"""Pool all two-hop question sets and simulate a size-aware router.

1) Pooled comparison over every two-hop question (multihop + coinvest + shared-investor):
   walk_40 vs expand_7k (similar token budget) with a paired bootstrap 95% CI.
2) Router simulation, no new runs needed: before answering, the agent can measure how big the
   2-hop neighbourhood is (one neighbour count query). Route to deterministic expansion when the
   whole neighbourhood fits the budget, otherwise to the random walk. We evaluate that rule on the
   per-question results that already exist.

    python scripts/two_hop_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TWO_HOP_KINDS = {"multihop", "coinvest", "sharedinv"}
SETS = {"explore": ROOT / "results/explore_budget/per_question.csv",
        "coinv": ROOT / "results/coinv_budget/per_question.csv"}
EXPAND_RUNS = {"explore": ROOT / "results/explore_expand_7000/raw.jsonl",
               "coinv": ROOT / "results/coinv_expand_7000/raw.jsonl"}
BUDGET = 7000


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return round(float(d.mean()), 3), round(float(lo), 3), round(float(hi), 3)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore-tag", default="explore", help="e.g. explore_final for the final-graph reruns")
    tag = ap.parse_args().explore_tag
    SETS["explore"] = ROOT / f"results/{tag}_budget/per_question.csv"
    EXPAND_RUNS["explore"] = ROOT / f"results/{tag}_expand_7000/raw.jsonl"
    frames = []
    for name, path in SETS.items():
        if not path.exists():
            sys.exit(f"missing {path}: run scripts/explore_budget.py for that set first")
        df = pd.read_csv(path)
        df["set"] = name
        frames.append(df[df.kind.isin(TWO_HOP_KINDS)])
    df = pd.concat(frames, ignore_index=True)

    # size of the full 2-hop neighbourhood per question (known before answering)
    size = {}
    for name, path in EXPAND_RUNS.items():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            size[(r["model"], r["qid"])] = ((r["metrics"].get("expand") or {}).get("est_full_tokens"))
    df["full_2hop_tokens"] = [size.get((m, q)) for m, q in zip(df.model, df.qid)]

    w = df.pivot_table(index=["model", "qid", "kind"], columns="method", values="recall").reset_index()
    t = df.pivot_table(index=["model", "qid"], columns="method", values="ctx_tokens").reset_index()
    s = df.drop_duplicates(["model", "qid"]).set_index(["model", "qid"])["full_2hop_tokens"]
    w = w.merge(t, on=["model", "qid"], suffixes=("", "_tok"))
    w["full_2hop_tokens"] = [s.get((m, q)) for m, q in zip(w.model, w.qid)]

    # router: expand when the whole 2-hop neighbourhood fits the budget, else walk
    small = w.full_2hop_tokens.fillna(np.inf) <= BUDGET
    w["router"] = np.where(small, w["expand_7k"], w["walk_40"])
    w["router_tok"] = np.where(small, w["expand_7k_tok"], w["walk_40_tok"])

    rows = []
    for model, g in w.groupby("model"):
        for label, gg in [("all two-hop", g)] + [(k, g[g.kind == k]) for k in sorted(g.kind.unique())]:
            r = {"model": model, "subset": label, "n": len(gg), "small_nbhd_share": round(float((gg.full_2hop_tokens <= BUDGET).mean()), 2)}
            for m in ["walk_40", "expand_7k", "router", "expand_16k"]:
                r[f"{m}_recall"] = round(float(gg[m].mean()), 3)
                r[f"{m}_tok"] = int(gg[f"{m}_tok"].mean())
            r["walk40-expand7k"] = boot((gg["walk_40"] - gg["expand_7k"]).to_numpy())
            r["router-expand7k"] = boot((gg["router"] - gg["expand_7k"]).to_numpy())
            r["router-walk40"] = boot((gg["router"] - gg["walk_40"]).to_numpy())
            rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    dest = ROOT / "results" / ("two_hop_summary.csv" if tag == "explore" else f"two_hop_summary_{tag}.csv")
    out.to_csv(dest, index=False)
    print(f"\n(mean diff, 95% CI low, high) from a paired bootstrap\nsaved -> {dest}")


if __name__ == "__main__":
    main()