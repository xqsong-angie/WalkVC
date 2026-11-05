"""Robust structured-output parsing + global success-rate bookkeeping.

The 'Reliability' metric in the proposal (>95% valid JSON/Cypher) is computed from
the counters kept here.
"""
from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_lock = threading.Lock()
_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "success": 0, "repaired": 0})


def record(kind: str, ok: bool, repaired: bool = False) -> None:
    with _lock:
        s = _stats[kind]
        s["attempts"] += 1
        s["success"] += int(ok)
        s["repaired"] += int(repaired)


def parser_stats(reset: bool = False) -> dict[str, dict[str, float]]:
    with _lock:
        out = {
            k: {**v, "rate": (v["success"] / v["attempts"]) if v["attempts"] else 0.0}
            for k, v in _stats.items()
        }
        if reset:
            _stats.clear()
    return out


_FENCE = re.compile(r"```(?:json|cypher)?\s*(.*?)```", re.S | re.I)


def _candidates(text: str):
    yield text.strip()
    for m in _FENCE.finditer(text):
        yield m.group(1).strip()
    # outermost {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            yield text[i : j + 1]


def _light_repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)          # trailing commas
    s = re.sub(r"(?<!\\)'", '"', s) if s.count('"') == 0 else s  # single quotes only
    return s


def parse_json(text: str, kind: str = "json") -> Any | None:
    """Return parsed JSON or None. Records success/failure under `kind`."""
    for i, cand in enumerate(_candidates(text)):
        for repaired, c in ((False, cand), (True, _light_repair(cand))):
            try:
                obj = json.loads(c)
                record(kind, True, repaired=repaired or i > 0)
                return obj
            except (json.JSONDecodeError, TypeError):
                continue
    record(kind, False)
    return None


def parse_model(text: str, model: Type[T], kind: str | None = None) -> T | None:
    kind = kind or model.__name__
    obj = parse_json(text, kind=f"{kind}:json")
    if obj is None:
        record(kind, False)
        return None
    try:
        m = model.model_validate(obj)
        record(kind, True)
        return m
    except ValidationError:
        record(kind, False)
        return None
