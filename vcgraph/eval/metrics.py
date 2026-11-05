"""Metric helpers for the four evaluation dimensions in the proposal.

Task quality : RAGAS (faithfulness -> hallucination rate, context precision, answer relevancy,
               factual correctness) + graph-level precision/recall against reference node ids
Latency      : TTFT, end-to-end, per-walk-iteration latency (target < 3 s)
Resource use : VRAM (Ollama /api/ps, nvidia-smi), total tokens, sampled-subgraph vs full-context tokens
Reliability  : structured-output parser success rate, Cypher execution success rate (target > 95%)
"""
from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np


def id_precision_recall(retrieved: list[str], reference: list[str], k: int | None = None) -> dict:
    ret = retrieved[:k] if k else retrieved
    ref = set(reference)
    if not ref:
        return {"precision": None, "recall": None}
    hits = sum(1 for r in ret if r in ref)
    return {"precision": hits / len(ret) if ret else 0.0, "recall": hits / len(ref)}


def cited_ids(answer: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([a-z]+:[^\]\s]+)\]", answer)))


def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return round(float(np.percentile(xs, q)), 4) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def nvidia_smi_used_mb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      text=True, timeout=5)
        return float(sum(float(x) for x in out.strip().splitlines()))
    except Exception:  # noqa: BLE001
        return None
