"""Ablation of the marginal-information-gain stopping rule.

Same walk (40 kept nodes), three stopping rules:
  info_gain (default)  WALK_GAIN_EPS=0.05  -> reuses coinv_walk_40; re-runs the explore set, because the
                                            explore runs were made before investor edges were added
  fixed_12             WALK_GAIN_EPS=0     -> gain is never < 0, so every walk runs all 12 iterations
  aggressive           WALK_GAIN_EPS=0.15  -> stops earlier
Question sets: the two-hop / exploration sets, where the walk matters.

    caffeinate -i python scripts/stopping_ablation.py            # run what's missing, then summarize
    python scripts/stopping_ablation.py --no-run                 # summarize only
Outputs results/stopping_ablation/{summary.csv, bootstrap.csv}.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
MODELS = ["llama3.1:8b", "qwen2.5:14b"]
SETS = {"explore": "data/yc/questions_explore.jsonl", "coinv": "data/yc/questions_coinvest.jsonl"}
RULES = {  # label -> (run-name suffix, env)
    "info_gain": ("walk40_gain", {"WALK_TOP_NODES": "40"}),
    "fixed_12": ("walk40_fixed", {"WALK_TOP_NODES": "40", "WALK_GAIN_EPS": "0"}),
    "aggressive": ("walk40_eps15", {"WALK_TOP_NODES": "40", "WALK_GAIN_EPS": "0.15"}),
}


def run_name(tag: str, suffix: str) -> str:
    # coinv_walk_40 already ran on the final graph with default settings -> reuse it
    return "coinv_walk_40" if (tag, suffix) == ("coinv", "walk40_gain") else f"{tag}_{suffix}"


def run_missing():
    for tag, qfile in SETS.items():
        for label, (suffix, env) in RULES.items():
            run = run_name(tag, suffix)
            if (ROOT / "results" / run / "raw.jsonl").exists():
                continue
            cmd = [sys.executable, "main.py", "eval", "--questions", qfile, "--models", *MODELS,
                   "--modes", "walk", "--no-ragas", "--run-name", run]
            print(f"\n>>> {run} ({label}) env={env}")
            subprocess.run(cmd, env={**os.environ, **env}, check=True)


def load():
    rows = []
    for tag in SETS:
        for label, (suffix, _) in RULES.items():
            path = ROOT / "results" / run_name(tag, suffix) / "raw.jsonl"
            if not path.exists():
                print(f"(missing {path.parent.name})")
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                m = r["metrics"]
                walks = m.get("walks") or []
                rows.append({"set": tag, "rule": label, "model": r["model"], "qid": r["qid"],
                             "kind": r["qid"].split("-")[0], "recall": r["retrieval_recall"],
                             "iterations": sum(w.get("iterations", 0) for w in walks),
                             "stopped_by_gain": float(np.mean([w.get("stop") == "info_gain" for w in walks])) if walks else None,
                             "walk_s": (m.get("timings") or {}).get("walk"),
                             "nodes_touched": sum(w.get("nodes_touched", 0) for w in walks),
                             "ctx_tokens": m.get("context_tokens"), "e2e_s": m.get("e2e_latency_s"),
                             "ttft_s": m.get("ttft_s")})
    return pd.DataFrame(rows)


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return round(float(d.mean()), 3), round(float(lo), 3), round(float(hi), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    if not ap.parse_args().no_run:
        run_missing()
    df = load()
    if df.empty:
        sys.exit("no results")
    df["rule"] = pd.Categorical(df["rule"], list(RULES), ordered=True)
    cols = ["recall", "iterations", "stopped_by_gain", "walk_s", "nodes_touched", "ctx_tokens", "e2e_s", "ttft_s"]
    summary = df.groupby(["model", "rule"], observed=True)[cols].mean().round(3)
    pd.set_option("display.width", 200)
    print("\n=== all questions ===")
    print(summary.to_string())
    print("\n=== by set ===")
    print(df.groupby(["set", "model", "rule"], observed=True)[["recall", "iterations", "walk_s", "nodes_touched"]]
          .mean().round(3).to_string())

    rows = []
    for model, g in df.groupby("model"):
        w = g.pivot_table(index="qid", columns="rule", values="recall", observed=True)
        for other in ["fixed_12", "aggressive"]:
            if other in w and "info_gain" in w:
                d = (w["info_gain"] - w[other]).dropna().to_numpy()
                rows.append({"model": model, "comparison": f"info_gain - {other}", "n": len(d),
                             "recall_diff (mean, lo, hi)": boot(d)})
    b = pd.DataFrame(rows)
    print("\n=== recall difference, paired bootstrap 95% CI ===")
    print(b.to_string(index=False))
    out = ROOT / "results" / "stopping_ablation"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv")
    b.to_csv(out / "bootstrap.csv", index=False)
    print(f"\nsaved -> {out}/summary.csv, bootstrap.csv")


if __name__ == "__main__":
    main()