"""Natural-language query decomposition into walk-able sub-queries."""
from __future__ import annotations

import re

from ..jsonparse import parse_model
from ..llm import Usage, chat
from ..schema import Decomposition, SubQuery, Sufficiency

LABELS = ["Company", "Investor", "Person", "Sector", "Batch", "Filing", "LegalEntity"]
RELS = ["FOUNDED", "INVESTED_IN", "IN_SECTOR", "IN_BATCH", "FILED", "RELATED_TO_FILING",
        "REGISTERED_AS", "SIMILAR_TO", "WORKED_AT"]

_SYS = f"""You plan exploration of a venture-capital knowledge graph.
Node labels: {LABELS}. Relationship types: {RELS}.
Decompose the user question into 1-3 focused sub-queries that can each be answered by exploring
the graph around a few seed entities. For each sub-query give:
  text            a short self-contained search phrase (include entity names / sector / product words)
  target_types    node labels that will hold the answer
  relation_hints  relationship types worth following
  anchors         exact entity names mentioned in the question (may be empty)
Also classify intent as one of: competitors, investors, founders, market_signal, lookup, other.
Return ONLY JSON: {{"intent": "...", "sub_queries": [{{"text": "...", "target_types": [], "relation_hints": [], "anchors": []}}]}}"""


def _fallback(question: str) -> Decomposition:
    q = question.lower()
    intent = ("competitors" if re.search(r"compet|rival|alternative|similar", q) else
              "investors" if re.search(r"investor|backed|fund|vc|co-invest", q) else
              "founders" if re.search(r"founder|founded|ceo|team|background", q) else
              "market_signal" if re.search(r"emerging|trend|recent|new|signal", q) else "other")
    anchors = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+)*)\b", question)
    anchors = [a for a in anchors if a.lower() not in {"which", "who", "what", "how", "the"}]
    return Decomposition(intent=intent, sub_queries=[SubQuery(text=question, anchors=anchors)])


def decompose(question: str, model: str | None, usage: Usage | None = None) -> tuple[Decomposition, bool]:
    """Returns (decomposition, parsed_ok)."""
    text, _ = chat([{"role": "system", "content": _SYS}, {"role": "user", "content": question}],
                   model, json_mode=True, max_tokens=400, purpose="decompose", usage=usage)
    d = parse_model(text, Decomposition, kind="decompose")
    if d is None or not d.sub_queries:
        return _fallback(question), False
    d.sub_queries = d.sub_queries[:3]
    for sq in d.sub_queries:
        sq.target_types = [t for t in sq.target_types if t in LABELS]
        sq.relation_hints = [r.upper() for r in sq.relation_hints if r.upper() in RELS]
    return d, True


_SUFF = """You check whether retrieved graph evidence is enough to answer a question.
Return ONLY JSON: {"sufficient": true|false, "missing": "<what is missing or null>",
"follow_up": "<a short search phrase to find the missing info, or null>"}"""


def check_sufficiency(question: str, context: str, model: str | None, usage: Usage | None = None) -> Sufficiency:
    text, _ = chat([{"role": "system", "content": _SUFF},
                    {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{context[:6000]}"}],
                   model, json_mode=True, max_tokens=150, purpose="sufficiency", usage=usage)
    return parse_model(text, Sufficiency, kind="sufficiency") or Sufficiency(sufficient=True)
