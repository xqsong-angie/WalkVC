"""Deterministic Text-to-Cypher baseline (what the random-walk agent is compared against).

Also used to measure 'Correct Cypher generation' and the Cypher execution success rate.
"""
from __future__ import annotations

import json
import re

from ..jsonparse import record
from ..llm import Usage, chat
from ..storage.graph import GraphBackend

_SYS = """You translate questions into a single READ-ONLY Neo4j 5 Cypher query.
Rules: use only the labels, relationship types and properties in the schema; match names
case-insensitively with toLower(n.name) CONTAINS toLower('<name>'); always RETURN readable
properties (name, one_liner, batch, total_raised, ...) and LIMIT 25. No CREATE/MERGE/SET/DELETE.
Return ONLY the Cypher query, no explanation.

SCHEMA:
{schema}"""


def _clean(text: str) -> str:
    m = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.S | re.I)
    q = (m.group(1) if m else text).strip().rstrip(";")
    return q


def generate_cypher(question: str, schema: str, model: str | None, usage: Usage | None = None,
                    error: str | None = None, previous: str | None = None) -> str:
    msgs = [{"role": "system", "content": _SYS.format(schema=schema)}, {"role": "user", "content": question}]
    if error and previous:
        msgs += [{"role": "assistant", "content": previous},
                 {"role": "user", "content": f"The query failed with: {error[:400]}\nReturn a corrected query."}]
    text, _ = chat(msgs, model, max_tokens=400, purpose="cypher", usage=usage)
    return _clean(text)


def cypher_retrieve(question: str, g: GraphBackend, model: str | None, usage: Usage | None = None,
                    retries: int = 1) -> dict:
    schema = g.schema_summary()
    q, err, rows = None, None, []
    first_ok = False
    for attempt in range(retries + 1):
        q = generate_cypher(question, schema, model, usage, error=err, previous=q)
        try:
            rows = g.run_cypher(q)
            record("cypher", True)
            first_ok = attempt == 0
            err = None
            break
        except Exception as e:  # noqa: BLE001  syntax error, write clause, unknown label...
            record("cypher", False)
            err = str(e)
    context = "CYPHER RESULT ROWS:\n" + "\n".join(json.dumps(r, default=str)[:400] for r in rows[:25])
    return {"cypher": q, "rows": rows, "error": err, "first_try_valid": first_ok,
            "non_empty": bool(rows), "context": context}
