"""LLM-based structured extraction (replaces hand-written scraping/parsing logic)."""
from __future__ import annotations

import json
import re
from ..config import settings
from ..jsonparse import parse_model
from ..llm import Usage, chat
from ..schema import CompanyProfile

_SYS = (
    "You extract structured startup data from a web page. Return ONLY a JSON object matching this "
    "JSON schema. Use null / [] when the page does not state a value. Never invent facts.\n"
    f"SCHEMA: {json.dumps(CompanyProfile.model_json_schema())}"
)


def _focus(markdown: str, window: int = 6000) -> str:
    """Keep only the useful part of a YC company page: the header and the founders section."""
    md = re.sub(r"\(https?://[^)\s]+\)", "", markdown)   # drop link targets
    md = re.sub(r"!\[[^\]]*\]", "", md)                    # drop images
    md = re.sub(r"\n{3,}", "\n\n", md)
    head = md[:2500]
    m = re.search(r"(?i)\bactive founders\b|\bfounders\b", md[2500:])
    start = 2500 + m.start() if m else 2500
    return head + "\n...\n" + md[start:start + window]


def extract_company(markdown: str, url: str, model: str | None = None,
                    usage: Usage | None = None, max_chars: int = 18000) -> CompanyProfile | None:
    model = model or settings.extract_model
    msgs = [{"role": "system", "content": _SYS},
                        {"role": "user", "content": f"URL: {url}\n\nPAGE (markdown):\n{_focus(markdown)[:max_chars]}"}]
    for attempt in range(2):
        text, _ = chat(msgs, model, json_mode=True, max_tokens=1200, purpose="extract", usage=usage)
        prof = parse_model(text, CompanyProfile, kind="extract")
        if prof:
            return prof
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": "That was not valid JSON for the schema. Return only the corrected JSON."}]
    return None


def profile_to_record(p: CompanyProfile, url: str) -> dict:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return {"source": "yc", "source_id": slug, "kind": "company", "name": p.name,
            "data": {**p.model_dump(), "yc_url": url,
                     "founders": [f.model_dump() for f in p.founders]}}
