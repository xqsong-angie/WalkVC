"""MongoDB raw/landing store.

Collections
-----------
raw_pages   : every fetched HTML / JSON / XML payload (source of truth, re-processable)
records     : per-source normalized records (SourceRecord)
entities    : resolved canonical entities after entity resolution
runs        : evaluation run logs
"""
from __future__ import annotations

import datetime as dt
import hashlib
from typing import Iterable

from pymongo import MongoClient, UpdateOne

from ..config import settings

_client: MongoClient | None = None


def db():
    global _client
    if _client is None:
        _client = MongoClient(settings.mongo_uri)
        d = _client[settings.mongo_db]
        d.raw_pages.create_index("url_hash", unique=True)
        d.records.create_index([("source", 1), ("source_id", 1)], unique=True)
        d.entities.create_index("entity_id", unique=True)
    return _client[settings.mongo_db]


def _now():
    return dt.datetime.now(dt.timezone.utc)


def save_raw(source: str, url: str, body: str, content_type: str = "text/html", meta: dict | None = None) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()
    db().raw_pages.update_one(
        {"url_hash": h},
        {"$set": {"source": source, "url": url, "body": body, "content_type": content_type,
                  "meta": meta or {}, "fetched_at": _now()}},
        upsert=True,
    )
    return h


def get_raw(url: str) -> dict | None:
    return db().raw_pages.find_one({"url_hash": hashlib.sha1(url.encode()).hexdigest()})


def save_records(records: Iterable[dict]) -> int:
    ops = [
        UpdateOne({"source": r["source"], "source_id": r["source_id"]},
                  {"$set": {**r, "updated_at": _now()}}, upsert=True)
        for r in records
    ]
    if not ops:
        return 0
    res = db().records.bulk_write(ops, ordered=False)
    return res.upserted_count + res.modified_count


def load_records(source: str | None = None) -> list[dict]:
    q = {"source": source} if source else {}
    return list(db().records.find(q, {"_id": 0}))


def save_entities(entities: list[dict]) -> None:
    ops = [UpdateOne({"entity_id": e["entity_id"]}, {"$set": e}, upsert=True) for e in entities]
    if ops:
        db().entities.bulk_write(ops, ordered=False)


def load_entities() -> list[dict]:
    return list(db().entities.find({}, {"_id": 0}))


def log_run(doc: dict) -> None:
    db().runs.insert_one({**doc, "ts": _now()})
