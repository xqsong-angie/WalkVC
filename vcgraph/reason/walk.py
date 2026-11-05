r"""Query-guided random walks with restart on the knowledge graph.

Instead of a deterministic Cypher query (or dumping a k-hop neighbourhood into the
prompt), W walkers sample paths from query-relevant seed nodes. The transition
probability from node u to neighbour v is

    P(v | u, q) ∝ exp( [ α·cos(q, e_v)                    semantic relevance to the (sub-)query
                        + β·log w_rel(type(u,v))           relation prior from query decomposition
                        + γ·log w_label(label(v))          target node-type prior
                        − λ·log(1 + deg(v))                hub penalty (avoid sector/batch super-nodes)
                        − μ·1[v == previous node] ] / T )  non-backtracking

with restart probability p_restart back to a seed (personalised-PageRank style).

All walkers advance synchronously, so each iteration costs two batched DB
round-trips (neighbours + missing embeddings) regardless of W. Only the touched
neighbourhood is ever materialised -> memory-efficient on large graphs.

Stopping: marginal information gain. Let the evidence set S_t be the top-K nodes
by importance. The information gained at iteration t is the relevance mass of the
nodes that *newly entered* the evidence set, relative to the whole set:

    IG_t = Σ_{n ∈ S_t \ S_{t−1}} rel⁺(n)  /  Σ_{n ∈ S_t} rel⁺(n)

We stop when IG_t < ε for `patience` consecutive iterations (the sampled subgraph
has stabilised — more walking would add little new evidence), or when max_iters /
the node budget is hit.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from ..storage.graph import GraphBackend, Nbr

INTENT_REL_PRIORS = {
    "competitors": {"SIMILAR_TO": 4.0, "IN_SECTOR": 2.0, "INVESTED_IN": 0.7, "FILED": 0.3},
    "investors": {"INVESTED_IN": 4.0, "IN_SECTOR": 1.5, "SIMILAR_TO": 1.0, "FILED": 0.5},
    "founders": {"FOUNDED": 4.0, "WORKED_AT": 3.0, "RELATED_TO_FILING": 1.5, "IN_SECTOR": 0.5},
    "market_signal": {"IN_BATCH": 2.5, "IN_SECTOR": 2.5, "FILED": 2.0, "SIMILAR_TO": 1.5},
}
INTENT_LABEL_PRIORS = {
    "competitors": {"Company": 2.5, "Sector": 1.0, "Filing": 0.3},
    "investors": {"Investor": 3.0, "Company": 1.2},
    "founders": {"Person": 3.0, "Company": 1.5},
    "market_signal": {"Company": 2.0, "Filing": 1.5, "Batch": 1.2},
}


@dataclass
class WalkParams:
    num_walkers: int = settings.walk_num_walkers
    max_iters: int = settings.walk_max_iters
    min_iters: int = 3
    restart_p: float = settings.walk_restart_p
    temperature: float = settings.walk_temperature
    gain_eps: float = settings.walk_gain_eps
    patience: int = settings.walk_patience
    top_k: int = settings.walk_top_nodes
    neighbor_cap: int = 64
    max_nodes_touched: int = 5000
    alpha: float = 1.0
    beta: float = 0.35
    gamma: float = 0.35
    lam: float = 0.15
    mu: float = 0.8
    seed: int = 0


@dataclass
class WalkResult:
    ranked: list[tuple[str, float]]                  # (node id, importance) top-K
    edges: list[tuple[str, str, str]]                # traversed edges among ranked nodes
    iterations: int
    stop_reason: str
    gains: list[float]
    iter_latency_s: list[float]
    db_roundtrips: int
    nodes_touched: int
    visits: dict[str, int] = field(default_factory=dict)
    relevance: dict[str, float] = field(default_factory=dict)


class RandomWalker:
    """Holds neighbour / embedding caches so several sub-queries share DB work."""

    def __init__(self, g: GraphBackend, params: WalkParams | None = None):
        self.g = g
        self.p = params or WalkParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.nbrs: dict[str, list[Nbr]] = {}
        self.emb: dict[str, np.ndarray] = {}
        self.meta: dict[str, tuple[str, int]] = {}   # id -> (label, degree)
        self.db_roundtrips = 0

    # ------------------------------------------------------------ cache fill
    def _ensure(self, ids: set[str]) -> None:
        need = [i for i in ids if i not in self.nbrs]
        if need:
            res = self.g.neighbors(need, cap=self.p.neighbor_cap)
            self.db_roundtrips += 1
            for i in need:
                self.nbrs[i] = res.get(i, [])
                for n in self.nbrs[i]:
                    self.meta.setdefault(n.id, (n.label, n.degree))
        missing = {n.id for i in ids for n in self.nbrs[i] if n.id not in self.emb} | {i for i in ids if i not in self.emb}
        if missing:
            got = self.g.get_embeddings(list(missing))
            self.db_roundtrips += 1
            for k, v in got.items():
                self.emb[k] = v / (np.linalg.norm(v) + 1e-9)
            for k in missing - got.keys():
                self.emb[k] = np.zeros(settings.embed_dim, dtype=np.float32)

    # ------------------------------------------------------------ main loop
    def run(self, q_vec: list[float], seeds: list[tuple[str, float]], intent: str = "other",
            rel_hints: list[str] | None = None, target_types: list[str] | None = None) -> WalkResult:
        p = self.p
        if not seeds:
            return WalkResult([], [], 0, "no_seeds", [], [], self.db_roundtrips, 0)
        q = np.asarray(q_vec, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9

        rel_w = dict(INTENT_REL_PRIORS.get(intent, {}))
        for r in rel_hints or []:
            rel_w[r.upper()] = max(rel_w.get(r.upper(), 1.0), 3.0)
        lab_w = dict(INTENT_LABEL_PRIORS.get(intent, {}))
        for t in target_types or []:
            lab_w[t] = max(lab_w.get(t, 1.0), 2.5)

        seed_ids = [s for s, _ in seeds]
        seed_logits = np.array([sc for _, sc in seeds], dtype=np.float64) / max(p.temperature, 1e-3)
        seed_p = np.exp(seed_logits - seed_logits.max())
        seed_p /= seed_p.sum()

        pos = self.rng.choice(seed_ids, size=p.num_walkers, p=seed_p).tolist()
        prev: list[str | None] = [None] * p.num_walkers
        visits: Counter = Counter(pos)
        edge_visits: Counter = Counter()
        rel_cache: dict[str, float] = {}

        def rel(i: str) -> float:
            if i not in rel_cache:
                rel_cache[i] = float(self.emb[i] @ q) if i in self.emb else 0.0
            return rel_cache[i]

        def evidence_set() -> set[str]:
            return {i for i, _ in self._rank(visits, rel, lab_w, k=p.top_k)}

        gains, lat = [], []
        prev_set = evidence_set()
        low = 0
        stop = "max_iters"
        it = 0
        for it in range(1, p.max_iters + 1):
            t0 = time.perf_counter()
            self._ensure(set(pos))
            new_pos, new_prev = [], []
            for w, (cur, pv) in enumerate(zip(pos, prev)):
                cand = self.nbrs.get(cur, [])
                if not cand or self.rng.random() < p.restart_p:
                    nxt = self.rng.choice(seed_ids, p=seed_p)
                    new_pos.append(str(nxt)); new_prev.append(None)
                    continue
                logits = np.array([
                    p.alpha * rel(n.id)
                    + p.beta * math.log(rel_w.get(n.rel, 1.0))
                    + p.gamma * math.log(lab_w.get(n.label, 1.0))
                    - p.lam * math.log1p(n.degree)
                    - (p.mu if n.id == pv else 0.0)
                    for n in cand
                ]) / p.temperature
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                n = cand[int(self.rng.choice(len(cand), p=probs))]
                edge_visits[(cur, n.rel, n.id) if n.outgoing else (n.id, n.rel, cur)] += 1
                new_pos.append(n.id); new_prev.append(cur)
            pos, prev = new_pos, new_prev
            visits.update(pos)
            # make sure newly reached nodes have embeddings before scoring them
            unseen = {i for i in pos if i not in self.emb}
            if unseen:
                got = self.g.get_embeddings(list(unseen)); self.db_roundtrips += 1
                for k in unseen:
                    v = got.get(k)
                    self.emb[k] = (v / (np.linalg.norm(v) + 1e-9)) if v is not None else np.zeros(settings.embed_dim, np.float32)

            cur_set = evidence_set()
            mass = sum(max(rel(i), 0.0) for i in cur_set)
            gain = sum(max(rel(i), 0.0) for i in cur_set - prev_set) / mass if mass > 1e-9 else 0.0
            prev_set = cur_set
            gains.append(round(gain, 4))
            lat.append(round(time.perf_counter() - t0, 4))
            if it >= p.min_iters:
                low = low + 1 if gain < p.gain_eps else 0
                if low >= p.patience:
                    stop = "info_gain"
                    break
            if len(self.nbrs) >= p.max_nodes_touched:
                stop = "node_budget"
                break

        ranked = self._rank(visits, rel, lab_w, k=p.top_k)
        top_ids = {i for i, _ in ranked}
        edges = [e for e in edge_visits if e[0] in top_ids and e[2] in top_ids]
        return WalkResult(ranked, edges, it, stop, gains, lat, self.db_roundtrips, len(self.nbrs),
                          visits={i: visits[i] for i in top_ids}, relevance={i: round(rel(i), 4) for i in top_ids})

    def _rank(self, visits: Counter, rel, lab_w: dict, k: int) -> list[tuple[str, float]]:
        if not visits:
            return []
        vmax = max(visits.values())
        scored = []
        for i, c in visits.items():
            label = self.meta.get(i, ("", 0))[0]
            imp = (0.5 * math.log1p(c) / math.log1p(vmax) + 0.5 * max(rel(i), 0.0)) * (1.25 if lab_w.get(label, 1.0) > 1.5 else 1.0)
            scored.append((i, round(imp, 4)))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------- context building
_SKIP = {"id", "label", "text", "embedding", "name_lc", "degree", "sources", "aliases", "url", "website"}


def serialize_subgraph(nodes: dict[str, dict], edges: list[tuple[str, str, str]], max_desc: int = 220) -> str:
    """Compact, citation-friendly textual rendering of the sampled subgraph."""
    lines = ["NODES:"]
    for nid, n in nodes.items():
        props = []
        for k, v in n.items():
            if k in _SKIP or k == "name" or v in (None, "", []):
                continue
            s = str(v)
            props.append(f"{k}={s[:max_desc]}")
        lines.append(f"[{nid}] {n.get('label')}: {n.get('name')}" + (" | " + "; ".join(props) if props else ""))
    if edges:
        lines.append("EDGES:")
        lines += [f"[{s}] -{t}-> [{d}]" for s, t, d in edges]
    return "\n".join(lines)


def full_context_baseline(g: GraphBackend, seed_ids: list[str], hops: int = 2, cap: int = 200) -> str:
    """Deterministic k-hop neighbourhood dump (the 'full context' baseline for token overhead)."""
    frontier, seen, edges = set(seed_ids), set(seed_ids), []
    for _ in range(hops):
        res = g.neighbors(list(frontier), cap=cap)
        nxt = set()
        for u, lst in res.items():
            for n in lst:
                edges.append((u, n.rel, n.id) if n.outgoing else (n.id, n.rel, u))
                if n.id not in seen:
                    seen.add(n.id); nxt.add(n.id)
        frontier = nxt
    return serialize_subgraph(g.get_nodes(list(seen)), sorted(set(edges)))
