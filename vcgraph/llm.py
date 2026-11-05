"""Unified LLM + embedding client.

Ollama exposes an OpenAI-compatible /v1 endpoint, so one SDK covers
Llama 3.1 8B, Qwen 2.5 14B (local) and GPT-4o (remote). Every call is
instrumented: TTFT, end-to-end latency, prompt/completion tokens.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Iterable

import httpx
import numpy as np
from openai import OpenAI

from .config import settings

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # offline / no cache
    _enc = None


def count_tokens(text: str) -> int:
    if _enc is None:
        return max(1, len(text) // 4)
    return len(_enc.encode(text, disallowed_special=()))


def is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4"))


_clients: dict[str, OpenAI] = {}


def client_for(model: str) -> OpenAI:
    key = "openai" if is_openai_model(model) else "ollama"
    if key not in _clients:
        if key == "openai":
            _clients[key] = OpenAI(api_key=settings.openai_api_key)
        else:
            _clients[key] = OpenAI(base_url=f"{settings.ollama_host}/v1", api_key="ollama")
    return _clients[key]


@dataclass
class CallStats:
    model: str
    purpose: str
    ttft_s: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Usage:
    """Accumulates per-query LLM usage (attach one per query run)."""

    calls: list[CallStats] = field(default_factory=list)

    def add(self, c: CallStats) -> None:
        self.calls.append(c)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    def summary(self) -> dict:
        return {
            "n_calls": len(self.calls),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "llm_time_s": round(sum(c.latency_s for c in self.calls), 3),
            "by_purpose": {
                p: round(sum(c.latency_s for c in self.calls if c.purpose == p), 3)
                for p in {c.purpose for c in self.calls}
            },
        }


def chat(
    messages: list[dict],
    model: str | None = None,
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    purpose: str = "chat",
    usage: Usage | None = None,
    on_token=None,
) -> tuple[str, CallStats]:
    """Streaming chat completion; returns (text, stats). TTFT is measured on first content delta."""
    model = model or settings.default_model
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens, stream=True)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if is_openai_model(model):
        kwargs["stream_options"] = {"include_usage": True}

    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    p_tok = c_tok = None
    for chunk in client_for(model).chat.completions.create(**kwargs):
        if getattr(chunk, "usage", None):
            p_tok = chunk.usage.prompt_tokens
            c_tok = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            delta = chunk.choices[0].delta.content
            parts.append(delta)
            if on_token:
                on_token(delta)
    latency = time.perf_counter() - t0
    text = "".join(parts)
    if p_tok is None:
        p_tok = sum(count_tokens(m.get("content", "")) for m in messages)
        c_tok = count_tokens(text)
    stats = CallStats(model, purpose, ttft if ttft is not None else latency, latency, p_tok, c_tok)
    if usage is not None:
        usage.add(stats)
    return text, stats


# ---------------------------------------------------------------- embeddings
def _hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic bag-of-words hashing embedding (offline tests only)."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def embed(texts: Iterable[str], model: str | None = None, batch: int = 64) -> list[list[float]]:
    texts = [t if t.strip() else "(empty)" for t in texts]
    if settings.embed_backend == "hash":
        return [_hash_embed(t, settings.embed_dim) for t in texts]
    model = model or settings.embed_model
    out: list[list[float]] = []
    cli = client_for(model)
    for i in range(0, len(texts), batch):
        resp = cli.embeddings.create(model=model, input=texts[i : i + batch])
        out.extend(d.embedding for d in resp.data)
    return out


# ---------------------------------------------------------------- VRAM
def ollama_vram() -> dict:
    """Ask Ollama which models are loaded and how much VRAM each uses."""
    try:
        r = httpx.get(f"{settings.ollama_host}/api/ps", timeout=3)
        r.raise_for_status()
        models = r.json().get("models", [])
        return {
            m["name"]: {"size_vram_mb": round(m.get("size_vram", 0) / 2**20, 1),
                        "size_mb": round(m.get("size", 0) / 2**20, 1)}
            for m in models
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
