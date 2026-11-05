import json
import os

os.environ["EMBED_BACKEND"] = "hash"

from vcgraph.collect import investors as inv  # noqa: E402
from vcgraph.llm import CallStats  # noqa: E402

YC_MD = """Reducto
Latest News
[Reducto: AI document parsing & extraction software](https://reducto.ai/blog/reducto-series-b-funding)
Oct 14, 2025
[Reducto: Making Human Data LLM-Ready](https://www.youtube.com/watch?v=QBC_cViA7j8)
May 01, 2025
[Exclusive: Reducto raises $24.5 million Series A led by Benchmark | Fortune](https://fortune.com/2025/04/25/x/)
Apr 25, 2025
[Reducto Document Ingestion API](https://reducto.ai/blog/seed-round)
Oct 02, 2024
[Reducto launches new API](https://reducto.ai/blog/new-api)
Sep 01, 2024
Company Launches
[See original launch post](https://www.ycombinator.com/launches/LYd)
"""


def test_parse_and_filter_yc_news():
    items = inv.parse_yc_news(YC_MD)
    assert len(items) == 5 and items[0]["date"] == "Oct 14, 2025"
    funding = [i["url"] for i in items if inv.is_funding_news(i)]
    assert funding == ["https://reducto.ai/blog/reducto-series-b-funding", "https://fortune.com/2025/04/25/x/",
                       "https://reducto.ai/blog/seed-round"]


def test_investor_key():
    k = inv.investor_key
    assert k("a16z") == k("Andreessen Horowitz") == k("Andreessen Horowitz (a16z)")
    assert k("Y Combinator") == k("YC")
    assert k("Khosla Ventures") == k("Khosla") and k("General Catalyst") == k("GC")
    assert k("Benchmark Capital") == k("Benchmark")
    assert inv.investor_node_id("Lightspeed Venture Partners") == inv.investor_node_id("Lightspeed")


def test_extract_and_merge(monkeypatch):
    def fake_chat(messages, model=None, **kw):
        out = {"about_company": True, "is_funding": True, "round": "Series A", "amount_usd": 24500000,
               "announced_date": "2025-04-25", "lead_investors": ["Benchmark"],
               "investors": ["Benchmark", "First Round Capital", "Reducto"]}
        return json.dumps(out), CallStats("fake", "x", 0, 0, 1, 1)
    monkeypatch.setattr(inv, "chat", fake_chat)
    company = {"entity_id": "co:reducto", "name": "Reducto", "attrs": {"one_liner": "document parsing"}}
    ev = inv.extract_event(company, {"title": "t", "url": "u"}, "The Series A was led by Benchmark, with First Round.")
    assert ev.round == "Series A" and ev.lead_investors == ["Benchmark"]
    rec = {"source": "news", "name": "Reducto", "data": {"entity_id": "co:reducto", "url": "u",
                                                        **ev.model_dump(exclude={"about_company", "is_funding"})}}
    ents, st = inv.merge_into_entities([{"entity_id": "co:reducto", "name": "Reducto", "investors": []}], [rec])
    names = {i["name"]: i for i in ents[0]["investors"]}
    assert set(names) == {"Benchmark", "First Round Capital"}            # company itself dropped
    assert names["Benchmark"]["lead"] and not names["First Round Capital"]["lead"]
    assert st["investor_edges"] == 2


def test_lead_verification_and_rounds():
    txt = "Navier raises $5.6M seed. The round was led by GV, with participation from HCVC and Y Combinator."
    assert inv._says_led("GV", txt) and not inv._says_led("HCVC", txt) and not inv._says_led("Y Combinator", txt)
    assert inv._says_led("Benchmark Capital", "Series A led by Benchmark")
    assert inv._says_led("Menlo Ventures", "Menlo Ventures led the round")
    assert inv._says_led("ICONIQ", "Our $80 million Series B led by ICONIQ and General Catalyst")
    assert inv._says_led("General Catalyst", "Our $80 million Series B led by ICONIQ and General Catalyst")
    assert not inv._says_led("Initialized Capital", "raises $4M from Initialized")
    assert [inv.normalize_round(x) for x in ["seed", "Series Unknown", "series b1", "Pre-seed round"]] == \
        ["Seed", None, "Series B1", "Pre-Seed"]