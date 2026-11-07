"""Recall-vs-token-budget comparison on the exploration set (open-ended + multi-hop questions).

1) Runs any missing experiment from RUNS below (only the 2-hop expansion at 7k tokens by default;
   the others were run before from the command line).
2) Prints recall / context tokens per method, overall and by question kind, and saves CSVs.
3) Paired bootstrap 95% CI for "walk_40 minus expand_7k" (same token budget) per model and kind.
4) Saves a recall-vs-context-tokens figure (results/explore_budget/recall_vs_tokens.png).

Usage (from the WalkVC folder, Docker + Ollama running):
    python scripts/explore_budget.py              # run what's missing, then compare
    python scripts/explore_budget.py --no-run     # only compare existing results
    python scripts/explore_budget.py --questions data/yc/questions_coinvest.jsonl --tag coinv   # another set
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

QUESTIONS = "data/yc/questions_explore.jsonl"
TAG = "explore"
MODELS = ["llama3.1:8b", "qwen2.5:14b"]


def make_runs(tag: str) -> dict:
    """run name -> (label for walk-mode rows, label for expand-mode rows, env needed to reproduce it)"""
    return {
        f"{tag}_anchor_only": ("anchor_only", None, {"WALK_MAX_ITERS": "0", "WALK_TOP_NODES": "12"}),
        f"{tag}_v1": ("walk_12", "expand_16k", {"WALK_TOP_NODES": "12", "EXPAND_MAX_TOKENS": "16000"}),
        f"{tag}_expand_2500": (None, "expand_2.5k", {"EXPAND_MAX_TOKENS": "2500"}),
        f"{tag}_expand_7000": (None, "expand_7k", {"EXPAND_MAX_TOKENS": "7000"}),
        f"{tag}_walk_40": ("walk_40", None, {"WALK_TOP_NODES": "40"}),
    }


RUNS = make_runs(TAG)
ORDER = ["anchor_only", "walk_12", "expand_2.5k", "expand_7k", "walk_40", "expand_16k"]


def run_missing():
    """Each experiment runs in its own Python process so its env settings take effect at import time."""
    import subprocess

    for run, (walk_label, expand_label, env) in RUNS.items():
        if (ROOT / "results" / run / "raw.jsonl").exists():
            continue
        modes = [m for m, lab in (("walk", walk_label), ("expand", expand_label)) if lab]
        modes_env = {**os.environ, **env}
        cmd = [sys.executable, "main.py", "eval", "--questions", QUESTIONS, "--models", *MODELS,
               "--modes", *modes, "--no-ragas", "--run-name", run]
        print(f"\n>>> running {run}  env={env}  modes={modes}")
        subprocess.run(cmd, env=modes_env, check=True)


def load_rows():
    rows = []
    for run, (walk_label, expand_label, _) in RUNS.items():
        path = ROOT / "results" / run / "raw.jsonl"
        if not path.exists():
            print(f"(missing: {run})")
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            label = walk_label if r["mode"] == "walk" else expand_label
            if not label:
                continue
            m = r["metrics"]
            rows.append({"model": r["model"], "method": label, "qid": r["qid"], "kind": r["qid"].split("-")[0],
                         "recall": r["retrieval_recall"], "ctx_tokens": m.get("context_tokens"),
                         "e2e_s": m.get("e2e_latency_s"), "ttft_s": m.get("ttft_s")})
    return rows


PAIRS = [("walk_40", "expand_7k"), ("walk_12", "expand_2.5k"), ("walk_40", "expand_16k")]


def paired_bootstrap(df, a, b, n_boot=10000, seed=0):
    """Mean recall difference a-b over questions answered by both, with a 95% percentile CI."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    out = []
    for (model, kind), g in pd.concat([df.assign(kind="all"), df]).groupby(["model", "kind"]):
        w = g.pivot_table(index="qid", columns="method", values="recall", observed=True)
        if a not in w or b not in w:
            continue
        d = (w[a] - w[b]).dropna().to_numpy()
        if not len(d):
            continue
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"model": model, "kind": kind, "comparison": f"{a} - {b}", "n": len(d),
                    "mean_diff": round(float(d.mean()), 3), "ci95_low": round(float(lo), 3),
                    "ci95_high": round(float(hi), 3), "wins": int((d > 0).sum()),
                    "ties": int((d == 0).sum()), "losses": int((d < 0).sum())})
    return pd.DataFrame(out)


def plot(df, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: pip install matplotlib  -> skipping the figure)")
        return
    families = {"walk": (["anchor_only", "walk_12", "walk_40"], "o-", "anchor + random walk"),
                "expand": (["expand_2.5k", "expand_7k", "expand_16k"], "s--", "deterministic 2-hop expansion")}
    models = sorted(df["model"].unique())
    kinds = ["all"] + sorted(df["kind"].unique())
    fig, axes = plt.subplots(len(models), len(kinds), figsize=(4.2 * len(kinds), 3.4 * len(models)),
                             squeeze=False, sharex=True, sharey=True)
    for i, model in enumerate(models):
        for j, kind in enumerate(kinds):
            ax = axes[i][j]
            sub = df[df.model == model] if kind == "all" else df[(df.model == model) & (df.kind == kind)]
            agg = sub.groupby("method", observed=True)[["recall", "ctx_tokens"]].mean()
            for methods, style, label in families.values():
                pts = agg.reindex([m for m in methods if m in agg.index]).sort_values("ctx_tokens")
                ax.plot(pts["ctx_tokens"], pts["recall"], style, label=label)
                for m, r in pts.iterrows():
                    ax.annotate(m, (r["ctx_tokens"], r["recall"]), fontsize=7,
                                textcoords="offset points", xytext=(4, -10))
            ax.set_title(f"{model} | {kind}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(models) - 1:
                ax.set_xlabel("context tokens given to the LLM")
            if j == 0:
                ax.set_ylabel("retrieval recall")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"figure -> {path}")


def main():
    global QUESTIONS, RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true", help="don't run missing experiments")
    ap.add_argument("--questions", default=QUESTIONS, help="question file to evaluate on")
    ap.add_argument("--tag", default=TAG, help="prefix of the run names, e.g. 'coinv' -> coinv_walk_40 ...")
    args = ap.parse_args()
    QUESTIONS, RUNS = args.questions, make_runs(args.tag)
    if not args.no_run:
        run_missing()

    import pandas as pd

    df = pd.DataFrame(load_rows())
    if df.empty:
        sys.exit("no results found")
    df["method"] = pd.Categorical(df["method"], [m for m in ORDER if m in set(df["method"])], ordered=True)
    cols = ["recall", "ctx_tokens", "e2e_s", "ttft_s"]

    overall = df.groupby(["model", "method"], observed=True)[cols].mean().round(3)
    by_kind = df.groupby(["kind", "model", "method"], observed=True)[cols].mean().round(3)
    pd.set_option("display.width", 160)
    print(f"\n=== overall ({df.groupby('model').qid.nunique().max()} questions per model) ===")
    print(overall.to_string())
    print("\n=== by question kind ===")
    print(by_kind.to_string())

    out = ROOT / "results" / f"{args.tag}_budget"
    out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out / "overall.csv")
    by_kind.to_csv(out / "by_kind.csv")
    df.to_csv(out / "per_question.csv", index=False)
    boot = pd.concat([paired_bootstrap(df, a, b) for a, b in PAIRS], ignore_index=True)
    print("\n=== paired bootstrap: recall difference (same questions), 95% CI ===")
    print(boot.to_string(index=False))
    boot.to_csv(out / "bootstrap.csv", index=False)
    plot(df, out / "recall_vs_tokens.png")
    print(f"\nsaved -> {out}/overall.csv, by_kind.csv, per_question.csv, bootstrap.csv")


if __name__ == "__main__":
    main()