"""Central configuration, loaded from .env / environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = _env("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = _env("MONGO_DB", "vcgraph")

    neo4j_uri: str = _env("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = _env("NEO4J_USER", "neo4j")
    neo4j_password: str = _env("NEO4J_PASSWORD", "password123")

    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    default_model: str = _env("DEFAULT_MODEL", "llama3.1:8b")
    extract_model: str = _env("EXTRACT_MODEL", "qwen2.5:14b")
    embed_model: str = _env("EMBED_MODEL", "nomic-embed-text")
    embed_dim: int = int(_env("EMBED_DIM", "768"))
    # "ollama" (real embeddings) or "hash" (deterministic offline embeddings for tests)
    embed_backend: str = _env("EMBED_BACKEND", "ollama")

    openai_api_key: str = _env("OPENAI_API_KEY", "")
    judge_model: str = _env("JUDGE_MODEL", "gpt-4o")

    sec_user_agent: str = _env("SEC_USER_AGENT", "VCGraph Research contact@example.com")
    opencorporates_token: str = _env("OPENCORPORATES_API_TOKEN", "")

    # Random-walk defaults (overridable per query)
    walk_num_walkers: int = int(_env("WALK_NUM_WALKERS", "32"))
    walk_max_iters: int = int(_env("WALK_MAX_ITERS", "12"))
    walk_restart_p: float = float(_env("WALK_RESTART_P", "0.15"))
    walk_temperature: float = float(_env("WALK_TEMPERATURE", "0.25"))
    walk_gain_eps: float = float(_env("WALK_GAIN_EPS", "0.05"))
    walk_patience: int = int(_env("WALK_PATIENCE", "2"))
    walk_top_nodes: int = int(_env("WALK_TOP_NODES", "20"))

    node_labels: tuple = field(
        default=("Company", "Investor", "Person", "Sector", "Batch", "Filing", "LegalEntity")
    )


settings = Settings()
