"""Anchoring / pinning / intersection on a small YC-shaped graph (offline)."""
import json
import os

os.environ["EMBED_BACKEND"] = "hash"

from vcgraph.kg.builder import build_graph  # noqa: E402
from vcgraph.llm import CallStats  # noqa: E402
from vcgraph.reason import agent as agent_mod  # noqa: E402
from vcgraph.reason import decompose as dec_mod  # noqa: E402
from vcgraph.reason.anchors import question_ngrams, resolve_anchors  # noqa: E402
from vcgraph.storage.graph import InMemoryGraph  # noqa: E402


def _ent(slug, name, batch, sectors, one_liner, founders=()):
    return {"entity_id": f"co:{slug}", "type": "company", "name": name, "sources": ["yc"], "aliases": [],
            "source_ids": [f"yc:{slug}"], "sectors": sectors, "investors": [], "filings": [], "legal_entities": [],
            "founders": [{"name": f, "title": "Founder"} for f in founders],
            "attrs": {"batch": batch, "one_liner": one_liner}}


def _graph():
    ents = [
        _ent("instinct", "Instinct", "Fall 2025", ["Consumer"], "AI pet health", ["Joon Won Song", "Eunju Moon"]),
        _ent("general-instinct", "General Instinct", "Fall 2025", ["Consumer"], "robot pets", ["Bill Jiao"]),
        _ent("selfin", "Selfin", "Fall 2025", ["Fintech"], "liquidity for prediction markets"),
        _ent("casey", "Casey", "Fall 2025", ["Fintech"], "AI bookkeeping"),
        _ent("amera", "Amera", "Fall 2025", ["Fintech"], "payments"),
        _ent("ledgerly", "Ledgerly", "Summer 2025", ["Fintech"], "AI bookkeeping"),
        _ent("toothy", "Toothy AI", "Winter 2025", ["Healthcare"], "AI front desk for dental clinics"),
        _ent("dodo", "Dodo", "Fall 2025", ["Healthcare"], "AI receptionist for dental clinics"),
    ]
    g = InMemoryGraph()
    build_graph(ents, g, verbose=False, sim_threshold=0.0)
    return g


def fake_chat(messages, model=None, *, json_mode=False, purpose="chat", usage=None, on_token=None, **kw):
    if purpose == "decompose":
        q = messages[-1]["content"]
        intent = "founders" if "founded" in q else "competitors" if "compete" in q else "market_signal"
        text = json.dumps({"intent": intent, "sub_queries": [{"text": q, "anchors": ["Distro"]}]})  # wrong LLM anchor
    else:
        text = "ok"
    st = CallStats("fake", purpose, 0.01, 0.01, 1, 1)
    if usage is not None:
        usage.add(st)
    return text, st


def test_ngrams_skip_stopwords():
    grams = question_ngrams("What Fintech startups are in the YC Fall 2025 batch?")
    assert "Fall 2025" in grams and "Fintech" in grams and "What" not in grams


def test_exact_anchor_beats_substring():
    g = _graph()
    names = [a["name"] for a in resolve_anchors(g, "Who founded Instinct?")]
    assert names == ["Instinct"]


def test_agent_pins(monkeypatch):
    g = _graph()
    monkeypatch.setattr(dec_mod, "chat", fake_chat)
    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    r = agent_mod.ask("Who founded Instinct?", g, model="fake", check=False)
    assert {"person:joon-won-song", "person:eunju-moon"} <= set(r["retrieved_ids"])
    r = agent_mod.ask("What Fintech startups are in the YC Fall 2025 batch?", g, model="fake", check=False)
    assert {"co:selfin", "co:casey", "co:amera"} <= set(r["retrieved_ids"])
    assert r["metrics"]["anchors"]["route"] == "intersection+walk"
    assert r["retrieved_ids"].index("co:selfin") < 6          # pinned, not buried
    r = agent_mod.ask("Which startups compete with Dodo?", g, model="fake", check=False)
    assert "co:dodo" in r["retrieved_ids"] and "co:toothy" in r["retrieved_ids"]




def _graph2():
    def e(slug, name, sectors, tags, founders):
        x = _ent(slug, name, "Fall 2025", sectors, f"{name} product", founders)
        x["attrs"]["tags"] = tags
        return x
    ents = [
        e("alpha", "Alpha", ["Healthcare"], ["Dental"], ["Jane Roe", "Max Doe"]),
        e("beta", "Beta", ["Fintech"], ["Payments"], ["Jane Roe"]),
        e("gamma", "Gamma", ["Healthcare"], ["Dental"], ["Ann Lee"]),
        e("delta", "Delta", ["Healthcare"], ["Dental"], ["Bo Kim"]),
        e("eps", "Eps", ["Fintech"], ["Payments"], ["Cy Wu"]),
    ]
    g = InMemoryGraph()
    build_graph(ents, g, verbose=False, sim_threshold=0.0)
    return g, ents


def test_new_question_kinds():
    from vcgraph.eval.make_questions import make_questions
    _, ents = _graph2()
    qs = {q["id"]: q for q in make_questions(ents, kinds={"open", "multihop"})}
    assert qs["multihop-alpha"]["reference_ids"] == ["co:beta"]
    assert set(qs["open-dental"]["reference_ids"]) == {"co:alpha", "co:gamma", "co:delta"}
    assert all(k.split("-")[0] in ("open", "multihop") for k in qs)


def test_expand_and_walk_reach_two_hops(monkeypatch):
    g, _ = _graph2()
    monkeypatch.setattr(dec_mod, "chat", fake_chat)
    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    q = "Which other YC companies share a founder or executive with Alpha?"
    r = agent_mod.ask(q, g, model="fake", mode="expand", check=False)
    assert "co:beta" in r["retrieved_ids"] and r["metrics"]["expand"]["total_nodes_2hop"] > 1
    r = agent_mod.ask(q, g, model="fake", mode="walk", check=False)
    assert "co:beta" in r["retrieved_ids"]