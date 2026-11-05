"""Offline tests: in-memory graph + hash embeddings + a scripted fake LLM.

Run:  EMBED_BACKEND=hash pytest -q
"""
from __future__ import annotations

import json
import os

os.environ["EMBED_BACKEND"] = "hash"

import pytest  # noqa: E402

from vcgraph.collect.resolve import normalize_name, resolve  # noqa: E402
from vcgraph.collect.sources import parse_form_d  # noqa: E402
from vcgraph.kg.builder import build_graph  # noqa: E402
from vcgraph.kg.demo import generate  # noqa: E402
from vcgraph.llm import CallStats  # noqa: E402
from vcgraph.reason import agent as agent_mod  # noqa: E402
from vcgraph.reason import decompose as dec_mod  # noqa: E402
from vcgraph.reason.walk import RandomWalker, WalkParams  # noqa: E402
from vcgraph.storage.graph import InMemoryGraph  # noqa: E402
from vcgraph.llm import embed  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    ents, qs = generate(n_companies=48, seed=3)
    g = InMemoryGraph()
    build_graph(ents, g, verbose=False)
    return g, ents, qs


def fake_chat(messages, model=None, *, json_mode=False, purpose="chat", usage=None, on_token=None, **kw):
    user = messages[-1]["content"]
    if purpose == "decompose":
        text = json.dumps({"intent": "competitors", "sub_queries": [
            {"text": user, "target_types": ["Company"], "relation_hints": ["SIMILAR_TO"], "anchors": []}]})
    elif purpose == "sufficiency":
        text = json.dumps({"sufficient": True, "missing": None, "follow_up": None})
    else:
        text = "Answer based on evidence [co:x]."
    st = CallStats(model or "fake", purpose, 0.01, 0.02, 10, 5)
    if usage is not None:
        usage.add(st)
    return text, st


def test_normalize_and_resolve():
    assert normalize_name("Acme Robotics, Inc.") == "acme robotics"
    recs = [
        {"source": "yc", "source_id": "acme", "kind": "company", "name": "Acme Robotics",
         "data": {"website": "https://acmerobotics.com", "sectors": ["Robotics"], "founders": [], "investors": []}},
        {"source": "sec", "source_id": "0001-24-1", "kind": "filing", "name": "ACME ROBOTICS INC",
         "data": {"company_hint": "Acme Robotics", "total_amount_sold": 2e6, "related_persons": []}},
        {"source": "opencorporates", "source_id": "us_de:123", "kind": "legal_entity", "name": "Acme Robotics Inc.",
         "data": {"company_hint": "Acme Robotics", "jurisdiction_code": "us_de"}},
        {"source": "yc", "source_id": "other", "kind": "company", "name": "Zephyr Grid", "data": {}},
    ]
    ents, st = resolve(recs)
    acme = next(e for e in ents if e["name"] == "Acme Robotics")
    assert set(acme["sources"]) == {"yc", "sec", "opencorporates"}
    assert len(ents) == 2 and st["multi_source_entities"] == 1


def test_parse_form_d():
    xml = """<edgarSubmission><primaryIssuer><cik>0001</cik><entityName>Acme Robotics Inc</entityName>
      <issuerAddress><city>SF</city><stateOrCountry>CA</stateOrCountry></issuerAddress></primaryIssuer>
      <relatedPersonsList><relatedPersonInfo><relatedPersonName><firstName>Ava</firstName><lastName>Arden</lastName>
      </relatedPersonName><relatedPersonRelationshipList><relationship>Executive Officer</relationship>
      </relatedPersonRelationshipList></relatedPersonInfo></relatedPersonsList>
      <offeringData><industryGroup><industryGroupType>Other Technology</industryGroupType></industryGroup>
      <typesOfSecuritiesOffered><isEquityType>true</isEquityType></typesOfSecuritiesOffered>
      <offeringSalesAmounts><totalOfferingAmount>5000000</totalOfferingAmount><totalAmountSold>3000000</totalAmountSold>
      </offeringSalesAmounts></offeringData></edgarSubmission>"""
    fd = parse_form_d(xml)
    assert fd["issuer_name"] == "Acme Robotics Inc" and fd["total_amount_sold"] == 3e6
    assert fd["related_persons"][0]["name"] == "Ava Arden" and fd["securities"] == ["isEquityType"]


def test_graph_built(demo):
    g, ents, _ = demo
    labels = {n["label"] for n in g.nodes.values()}
    assert {"Company", "Investor", "Person", "Sector", "Batch", "Filing"} <= labels
    assert any(t == "SIMILAR_TO" for _, t, _ in g.edges)


def test_walk_stops_and_ranks(demo):
    g, ents, qs = demo
    q = next(q for q in qs if q["type"] == "investors")
    v = embed([q["question"]])[0]
    seeds = [(n["id"], s) for n, s in g.vector_search(v, 5)]
    w = RandomWalker(g, WalkParams(num_walkers=16, max_iters=15, seed=1))
    res = w.run(v, seeds, intent="investors")
    assert res.ranked and res.iterations <= 15
    assert res.stop_reason in {"info_gain", "max_iters", "node_budget"}
    assert len(res.gains) == res.iterations
    assert max(res.iter_latency_s) < 3.0
    # sampled neighbourhood must be much smaller than the graph
    assert res.nodes_touched < len(g.nodes)


def test_agent_end_to_end(demo, monkeypatch):
    g, _, qs = demo
    monkeypatch.setattr(dec_mod, "chat", fake_chat)
    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    r = agent_mod.ask(qs[0]["question"], g, model="fake", check=True)
    assert r["answer"] and r["contexts"] and r["retrieved_ids"]
    m = r["metrics"]
    assert m["parse_ok"]["decompose"] is True
    assert m["context_tokens"] > 0 and m["walks"][0]["iterations"] >= 2


# ------------------------------------------------------------- real-data (YC JSON shape) path
def _yc_rows():
    base = {"industry": "B2B", "batch": "Winter 2025", "status": "Active", "team_size": 5,
            "all_locations": "San Francisco, CA, USA"}
    return [
        {**base, "slug": "alpha-ledger", "name": "Alpha Ledger", "website": "https://alphaledger.example",
         "one_liner": "AI bookkeeping for startups", "subindustry": "B2B -> Finance and Accounting",
                  "tags": ["Fintech", "AI", "Accounting", "Bookkeeping"], "industries": ["B2B", "Fintech"],
         "url": "https://www.ycombinator.com/companies/alpha-ledger"},
        {**base, "slug": "beta-books", "name": "Beta Books", "website": "https://betabooks.example",
         "one_liner": "Automated close for finance teams", "subindustry": "B2B -> Finance and Accounting",
         "tags": ["Fintech", "Accounting", "Bookkeeping"], "industries": ["B2B"], "batch": "W25",
         "url": "https://www.ycombinator.com/companies/beta-books"},
        {**base, "slug": "gamma-ops", "name": "Gamma Ops", "website": "https://gammaops.example",
         "one_liner": "Incident response copilot", "subindustry": "B2B -> Engineering",
         "tags": ["DevOps"], "industries": ["B2B"],
         "url": "https://www.ycombinator.com/companies/gamma-ops"},
    ]


def test_normalize_batch():
    from vcgraph.collect.sources import normalize_batch
    assert normalize_batch("W25") == normalize_batch("winter-2025") == "Winter 2025"
    assert normalize_batch("s24") == "Summer 2024" and normalize_batch("Fall 2024") == "Fall 2024"


def test_real_data_path_to_questions():
    from vcgraph.collect.sources import yc_oss_to_record
    from vcgraph.eval.make_questions import make_questions

    recs = [yc_oss_to_record(r) for r in _yc_rows()]
    recs[0]["data"]["founders"] = [{"name": "Jane Roe", "title": "CEO"}]   # as if crawled (hybrid mode)
    recs[1]["data"]["founders"] = [{"name": "Jane Roe", "title": "Advisor"}]
    ents, _ = resolve(recs)
    g = InMemoryGraph()
    build_graph(ents, g, verbose=False)
    assert g.nodes["co:alpha-ledger"]["subindustry"] == "B2B -> Finance and Accounting"
    assert "batch:winter-2025" in g.nodes                                 # W25 and Winter 2025 unified
    qs = {q["id"]: q for q in make_questions(ents)}
    assert qs["comp-alpha-ledger"]["reference_ids"] == ["co:beta-books"]
    assert qs["founder-alpha-ledger"]["reference_ids"] == ["person:jane-roe"]
    assert set(qs["network-jane-roe"]["reference_ids"]) == {"co:alpha-ledger", "co:beta-books"}
    assert any(q.startswith("emerging-b2b-winter-2025") for q in qs)
    # every reference id must exist in the graph
    assert all(i in g.nodes for q in qs.values() for i in q["reference_ids"])
