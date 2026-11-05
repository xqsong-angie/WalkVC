"""Graph Reasoning Agent (LangGraph).

    ┌──────────┐   ┌──────┐   ┌──────┐   ┌──────────┐   ┌───────┐   ┌────────┐
    │decompose │──▶│ seed │──▶│ walk │──▶│ context  │──▶│ check │──▶│ answer │
    └──────────┘   └──────┘   └──────┘   └──────────┘   └───┬───┘   └────────┘
                       ▲                                    │ insufficient → follow-up sub-query
                       └────────────────────────────────────┘
    mode="cypher": decompose is skipped, a Text-to-Cypher baseline retrieves instead of walks.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..llm import Usage, chat, count_tokens, embed
from ..schema import SubQuery
from ..storage.graph import GraphBackend
from .anchors import anchor_pins, condition_on_anchor, resolve_anchors
from .cypher import cypher_retrieve
from .decompose import check_sufficiency, decompose
from .walk import RandomWalker, WalkParams, serialize_subgraph

ANSWER_SYS = """You are a venture-capital market-intelligence analyst.
Answer the question using ONLY the knowledge-graph evidence provided. Cite the node ids you rely on
in square brackets, e.g. [co:nimbusly]. If the evidence does not contain the answer, say so plainly
rather than guessing. Be concise: a short paragraph or a bullet list."""


@dataclass
class Runtime:
    g: GraphBackend
    walker: RandomWalker
    usage: Usage = field(default_factory=Usage)
    timings: dict[str, float] = field(default_factory=dict)
    walk_traces: list[dict] = field(default_factory=list)
    parse_ok: dict[str, bool] = field(default_factory=dict)
    cypher: dict | None = None
    anchor_info: dict = field(default_factory=dict)
    expand_info: dict | None = None

    def tick(self, name: str, t0: float):
        self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - t0, 4)


class State(TypedDict, total=False):
    rt: Runtime
    question: str
    model: str
    mode: str
    check: bool
    max_rounds: int
    intent: str
    sub_queries: list[SubQuery]
    pending: list[int]
    seeds: dict[int, list[tuple[str, float]]]
    ranked: dict[str, float]
    edges: list[tuple[str, str, str]]
    round: int
    context: str
    contexts: list[str]
    answer: str
    ttft_s: float
    sq_vecs: dict[int, list[float]]
    pins: dict[str, float]
    anchors: list[dict]
    context_ids: list[str]
    on_token: Any


# ---------------------------------------------------------------- nodes
def n_decompose(s: State) -> State:
    t0 = time.perf_counter()
    d, ok = decompose(s["question"], s["model"], s["rt"].usage)
    s["rt"].parse_ok["decompose"] = ok
    s["rt"].tick("decompose", t0)
    return {"intent": d.intent, "sub_queries": d.sub_queries, "pending": list(range(len(d.sub_queries))),
            "round": 0, "ranked": {}, "edges": [], "seeds": {}}


def n_seed(s: State) -> State:
    t0 = time.perf_counter()
    rt, seeds, sq_vecs = s["rt"], dict(s.get("seeds", {})), dict(s.get("sq_vecs", {}))
    intent = s.get("intent", "other")
    out: State = {}
    anchors = s.get("anchors")
    if anchors is None:                                          # first round: ground the question
        llm_anchors = [a for sq in s["sub_queries"] for a in sq.anchors]
        anchors = resolve_anchors(rt.g, s["question"], llm_anchors)
        targets = {t for sq in s["sub_queries"] for t in sq.target_types}
        pins, info = anchor_pins(rt.g, anchors, intent, targets)
        rt.anchor_info = {**info, "anchor_names": [a["name"] for a in anchors], "n_pinned": len(pins)}
        out.update(anchors=anchors, pins=pins)
    for a in anchors:
        rt.walker.meta.setdefault(a["id"], (a["label"], 1))

    sqs = [s["sub_queries"][i] for i in s["pending"]]
    vecs = embed([sq.text for sq in sqs])
    for i, sq, v in zip(s["pending"], sqs, vecs):
        if intent == "competitors":
            v = condition_on_anchor(v, rt.g, anchors)
        found: dict[str, float] = {a["id"]: 1.0 for a in anchors}  # exact anchors get top weight
        for n in rt.g.find_by_name(sq.anchors, exact_only=True):
            found.setdefault(n["id"], 1.0)
            rt.walker.meta.setdefault(n["id"], (n["label"], 1))
        for n, sc in rt.g.vector_search(v, k=6):
            found.setdefault(n["id"], sc)
            rt.walker.meta.setdefault(n["id"], (n["label"], 1))
        seeds[i] = sorted(found.items(), key=lambda x: -x[1])[:8]
        sq_vecs[i] = v
    rt.tick("seed", t0)
    return {**out, "seeds": seeds, "sq_vecs": sq_vecs}


def n_walk(s: State) -> State:
    t0 = time.perf_counter()
    rt = s["rt"]
    ranked, edges = dict(s.get("ranked", {})), list(s.get("edges", []))
    for i in s["pending"]:
        sq = s["sub_queries"][i]
        res = rt.walker.run(s["sq_vecs"][i], s["seeds"][i], intent=s.get("intent", "other"),
                            rel_hints=sq.relation_hints, target_types=sq.target_types)
        for nid, imp in res.ranked:
            ranked[nid] = max(ranked.get(nid, 0.0), imp)
        edges.extend(e for e in res.edges if e not in edges)
        rt.walk_traces.append({"sub_query": sq.text, "iterations": res.iterations, "stop": res.stop_reason,
                               "gains": res.gains, "iter_latency_s": res.iter_latency_s,
                               "nodes_touched": res.nodes_touched, "top": res.ranked[:10]})
    rt.tick("walk", t0)
    return {"ranked": ranked, "edges": edges}


def n_context(s: State) -> State:
    t0 = time.perf_counter()
    rt = s["rt"]
    pins = s.get("pins", {})
    pinned = [i for i, _ in sorted(pins.items(), key=lambda x: -x[1])]
    walked = [i for i, _ in sorted(s["ranked"].items(), key=lambda x: -x[1]) if i not in pins]
    ids = pinned + walked[: max(settings.walk_top_nodes + 5 - len(pinned), 5)]
    nodes = rt.g.get_nodes(ids)
    nodes = {i: nodes[i] for i in ids if i in nodes}
    edges = [e for e in s["edges"] if e[0] in nodes and e[2] in nodes]
    edges += [e for e in rt.g.edges_among(ids) if e not in edges]
    ctx = serialize_subgraph(nodes, edges)
    rt.tick("context", t0)
    return {"context": ctx, "contexts": _chunks(ctx), "context_ids": list(nodes)}


EXPAND_MAX_TOKENS = int(os.getenv("EXPAND_MAX_TOKENS", "16000"))


def n_expand(s: State) -> State:
    """Baseline: deterministic 2-hop expansion around the anchors (no sampling), BFS order,
    truncated to a token budget. Shows what the walk saves on multi-hop questions."""
    t0 = time.perf_counter()
    rt = s["rt"]
    roots = [a["id"] for a in s.get("anchors", [])] or [i for v in s.get("seeds", {}).values() for i, _ in v[:3]]
    roots = list(dict.fromkeys(roots))
    order, seen, edges, frontier = list(roots), set(roots), [], roots
    for cap in (500, 100):                                   # hop 1, hop 2
        res = rt.g.neighbors(frontier, cap=cap) if frontier else {}
        nxt = []
        for u in frontier:
            for n in res.get(u, []):
                edges.append((u, n.rel, n.id) if n.outgoing else (n.id, n.rel, u))
                if n.id not in seen:
                    seen.add(n.id)
                    order.append(n.id)
                    nxt.append(n.id)
        frontier = nxt
    nodes = rt.g.get_nodes(order[:3000])
    kept, used = [], 0
    for i in order[:3000]:
        if i not in nodes:
            continue
        t = count_tokens(serialize_subgraph({i: nodes[i]}, []))
        if used + t > EXPAND_MAX_TOKENS:
            break
        kept.append(i)
        used += t
    keep = set(kept)
    edge_lines = [e for e in dict.fromkeys(edges) if e[0] in keep and e[2] in keep]
    ctx = serialize_subgraph({i: nodes[i] for i in kept}, [])
    budget_left = EXPAND_MAX_TOKENS - count_tokens(ctx)
    if edge_lines and budget_left > 0:
        per_edge = max(1, count_tokens(serialize_subgraph({}, edge_lines[:50])) // min(50, len(edge_lines)))
        edge_lines = edge_lines[: budget_left // per_edge]
    ctx = serialize_subgraph({i: nodes[i] for i in kept}, edge_lines)
    rt.expand_info = {"total_nodes_2hop": len(order), "included_nodes": len(kept),
                      "truncated": len(kept) < len(order),
                      "est_full_tokens": int(used / max(len(kept), 1) * len(order))}
    rt.tick("expand", t0)
    return {"context": ctx, "contexts": _chunks(ctx), "context_ids": kept}

def n_check(s: State) -> State:
    if not s.get("check") or s.get("round", 0) >= s.get("max_rounds", 1):
        return {"pending": []}
    t0 = time.perf_counter()
    suff = check_sufficiency(s["question"], s["context"], s["model"], s["rt"].usage)
    s["rt"].tick("check", t0)
    if suff.sufficient or not suff.follow_up:
        return {"pending": []}
    sqs = list(s["sub_queries"]) + [SubQuery(text=suff.follow_up)]
    return {"sub_queries": sqs, "pending": [len(sqs) - 1], "round": s.get("round", 0) + 1}


def n_cypher(s: State) -> State:
    t0 = time.perf_counter()
    res = cypher_retrieve(s["question"], s["rt"].g, s["model"], s["rt"].usage)
    s["rt"].cypher = {k: v for k, v in res.items() if k != "rows"} | {"n_rows": len(res["rows"])}
    s["rt"].tick("cypher", t0)
    return {"context": res["context"], "contexts": _chunks(res["context"])}


def n_answer(s: State) -> State:
    t0 = time.perf_counter()
    text, st = chat([{"role": "system", "content": ANSWER_SYS},
                     {"role": "user", "content": f"QUESTION: {s['question']}\n\nEVIDENCE:\n{s['context']}"}],
                    s["model"], max_tokens=600, purpose="answer", usage=s["rt"].usage,
                    on_token=s.get("on_token"))
    s["rt"].tick("answer", t0)
    return {"answer": text, "ttft_s": round(st.ttft_s, 4)}


def _chunks(ctx: str) -> list[str]:
    return [ln for ln in ctx.splitlines() if ln and not ln.endswith(":")]


def _route_start(s: State) -> str:
    return "cypher" if s.get("mode") == "cypher" else "decompose"


def _route_seed(s: State) -> str:
    return "expand" if s.get("mode") == "expand" else "walk"


def _route_check(s: State) -> str:
    return "seed" if s.get("pending") else "answer"


def build_agent():
    sg = StateGraph(State)
    for name, fn in [("decompose", n_decompose), ("seed", n_seed), ("walk", n_walk), ("context", n_context),
                                          ("check", n_check), ("cypher", n_cypher), ("expand", n_expand), ("answer", n_answer)]:
        sg.add_node(name, fn)
    sg.add_conditional_edges(START, _route_start, {"cypher": "cypher", "decompose": "decompose"})
    sg.add_edge("decompose", "seed")
    sg.add_conditional_edges("seed", _route_seed, {"walk": "walk", "expand": "expand"})
    sg.add_edge("expand", "answer")
    sg.add_edge("walk", "context")
    sg.add_edge("context", "check")
    sg.add_conditional_edges("check", _route_check, {"seed": "seed", "answer": "answer"})
    sg.add_edge("cypher", "answer")
    sg.add_edge("answer", END)
    return sg.compile()


_AGENT = None


def ask(question: str, g: GraphBackend, model: str | None = None, mode: str = "walk",
        check: bool = True, max_rounds: int = 1, walk_params: WalkParams | None = None,
        on_token=None) -> dict[str, Any]:
    """Run one query end-to-end and return answer + full telemetry."""
    global _AGENT
    _AGENT = _AGENT or build_agent()
    model = model or settings.default_model
    rt = Runtime(g=g, walker=RandomWalker(g, walk_params))
    t0 = time.perf_counter()
    out = _AGENT.invoke({"rt": rt, "question": question, "model": model, "mode": mode,
                         "check": check, "max_rounds": max_rounds, "on_token": on_token},
                        {"recursion_limit": 40})
    e2e = time.perf_counter() - t0
    return {
        "question": question, "model": model, "mode": mode,
        "intent": out.get("intent"),
        "sub_queries": [sq.text for sq in out.get("sub_queries", [])],
        "answer": out.get("answer", ""),
        "contexts": out.get("contexts", []),
        "retrieved_ids": out.get("context_ids", []),          # exactly what the answering LLM saw
        "seed_ids": sorted({i for v in out.get("seeds", {}).values() for i, _ in v}),
        "metrics": {
            "e2e_latency_s": round(e2e, 4),
            "ttft_s": out.get("ttft_s"),
            "timings": rt.timings,
            "context_tokens": count_tokens(out.get("context", "")),
            "llm": rt.usage.summary(),
            "db_roundtrips": rt.walker.db_roundtrips,
            "walks": rt.walk_traces,
            "parse_ok": rt.parse_ok,
            "cypher": rt.cypher,
            "anchors": rt.anchor_info,
            "expand": rt.expand_info,
        },
    }