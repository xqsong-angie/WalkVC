"""Anchoring: ground the question in exact graph nodes before (and alongside) the random walk.

Findings from the first real-data evaluation (yc_v1) that motivated this module:
  * Entity questions failed when the named company never became a seed ("Who founded Instinct?"
    seeded "General Instinct"), and the LLM decomposition sometimes rewrote names ("Dodo" -> "Distro").
    -> anchors are found by exact name match of question n-grams, independent of the LLM.
  * "Which startups compete with X?" embeds as generic text with no notion of what X does.
    -> for competitor questions the query vector is conditioned on X's own embedding.
  * "Which Fintech startups are in YC Fall 2025?" is a set intersection (batch AND sector).
    A sampling walk cannot enumerate it; an exact intersection over the two anchors can.
Anchored nodes and their direct, type-relevant neighbours are *pinned* into the evidence set.
"""
from __future__ import annotations

import re

import numpy as np

from ..storage.graph import GraphBackend

STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "by", "is", "are", "was", "has", "have",
    "which", "who", "what", "how", "much", "many", "its", "their", "they", "them", "it", "be", "been", "from",
    "startups", "startup", "companies", "company", "compete", "competes", "competitors", "founded", "founder",
    "founders", "connected", "raised", "according", "sec", "form", "filings", "filing", "backed", "investors",
    "investor", "building", "build", "yc", "batch", "most", "recent", "active", "emerging", "early", "stage",
    "background", "alongside", "co-invested", "invested", "same", "similar", "like", "do", "does", "did",
}
INTENT_TARGETS = {
    "founders": {"Person", "Company"},
    "investors": {"Investor"},
    "competitors": {"Company"},
    "lookup": {"Filing", "Person", "Investor", "LegalEntity"},
    "market_signal": {"Company", "Filing"},
}
STRUCTURAL = {"Batch", "Sector"}


def question_ngrams(question: str, max_n: int = 5) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9.&'+\-]*", question)
    toks = [t.rstrip(".'") for t in toks]
    out = []
    for n in range(max_n, 0, -1):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if all(t.lower() in STOP for t in gram):
                continue
            if n == 1 and (gram[0].lower() in STOP or len(gram[0]) < 2):
                continue
            out.append(" ".join(gram))
    return list(dict.fromkeys(out))


def resolve_anchors(g: GraphBackend, question: str, llm_anchors: list[str] | None = None) -> list[dict]:
    """Exact-name anchors from question n-grams (LLM-independent) plus LLM-proposed anchors."""
    cands = question_ngrams(question) + [a for a in (llm_anchors or []) if a]
    found = g.find_by_name(cands, exact_only=True)
    # prefer the longest match: drop "Fall" if "Fall 2025" also matched, drop "Labs" if "Haplotype Labs" matched
    names = [f["name"].lower() for f in found]
    keep = [f for f in found
            if not any(f["name"].lower() != o and re.search(rf"\b{re.escape(f['name'].lower())}\b", o) for o in names)]
    return keep


def anchor_pins(g: GraphBackend, anchors: list[dict], intent: str, target_types: set[str] | None = None,
                per_anchor: int = 10, max_intersection: int = 30) -> tuple[dict[str, float], dict]:
    """Return ({node_id: score}, info). Anchors get 1.0, their typed neighbours 0.9,
    exact batch/sector intersections 0.95."""
    pins: dict[str, float] = {a["id"]: 1.0 for a in anchors}
    info: dict = {"anchors": [a["id"] for a in anchors], "route": "walk"}
    targets = set(target_types or ()) | INTENT_TARGETS.get(intent, set())

    structural = [a for a in anchors if a["label"] in STRUCTURAL]
    if len(structural) >= 2:                               # e.g. Batch "Fall 2025" AND Sector "Fintech"
        hits = g.common_neighbors([a["id"] for a in structural], "Company", limit=max_intersection)
        for h in hits:
            pins[h["id"]] = 0.95
        info.update(route="intersection+walk", intersection=len(hits))

    entity_anchors = [a for a in anchors if a["label"] not in STRUCTURAL]
    if entity_anchors and targets:
        nb = g.neighbors([a["id"] for a in entity_anchors], cap=200)
        for a in entity_anchors:
            typed = [n for n in nb.get(a["id"], []) if n.label in targets and n.label not in STRUCTURAL]
            # competitor questions: follow the similarity edges of the anchor company first
            typed.sort(key=lambda n: (n.rel != "SIMILAR_TO", n.rel != "FOUNDED"))
            for n in typed[:per_anchor]:
                pins.setdefault(n.id, 0.9)
    return pins, info


def condition_on_anchor(q_vec, g: GraphBackend, anchors: list[dict], weight: float = 0.7):
    """Competitor questions: what X *does* matters, not the words 'compete with X'."""
    cos = [a["id"] for a in anchors if a["label"] == "Company"]
    if not cos:
        return q_vec
    embs = g.get_embeddings(cos[:1])
    if not embs:
        return q_vec
    a = next(iter(embs.values()))
    q = np.asarray(q_vec, dtype=np.float32)
    v = (1 - weight) * q / (np.linalg.norm(q) + 1e-9) + weight * a / (np.linalg.norm(a) + 1e-9)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()