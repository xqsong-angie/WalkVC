"""Graph backends.

Neo4jGraph   : production backend (Neo4j 5.x holds both KG structure and vector index).
InMemoryGraph: dependency-free backend with the same interface, used for unit tests
               and quick experiments without Docker.

The random-walk engine only needs: vector_search, find_by_name, neighbors,
get_embeddings, get_nodes. It never pulls the full graph into memory — only the
neighbourhoods actually touched by walkers (memory-efficient sampling).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import settings

LABELS = set(settings.node_labels)
REL_TYPES = {
    "FOUNDED", "INVESTED_IN", "IN_SECTOR", "IN_BATCH", "FILED",
    "RELATED_TO_FILING", "REGISTERED_AS", "SIMILAR_TO", "WORKED_AT",
}


@dataclass
class Nbr:
    rel: str
    outgoing: bool
    id: str
    label: str
    name: str
    degree: int


class GraphBackend(Protocol):
    def ensure_schema(self) -> None: ...
    def upsert_nodes(self, label: str, rows: list[dict]) -> None: ...
    def upsert_edges(self, rel: str, rows: list[dict]) -> None: ...
    def finalize(self) -> None: ...
    def vector_search(self, vec: list[float], k: int, labels: list[str] | None = None) -> list[tuple[dict, float]]: ...
    def find_by_name(self, names: list[str], exact_only: bool = False) -> list[dict]: ...
    def common_neighbors(self, ids: list[str], label: str, limit: int = 50) -> list[dict]: ...
    def neighbors(self, ids: list[str], cap: int = 64) -> dict[str, list[Nbr]]: ...
    def get_embeddings(self, ids: list[str]) -> dict[str, np.ndarray]: ...
    def get_nodes(self, ids: list[str]) -> dict[str, dict]: ...
    def edges_among(self, ids: list[str]) -> list[tuple[str, str, str]]: ...
    def schema_summary(self) -> str: ...
    def run_cypher(self, query: str, params: dict | None = None, limit: int = 50) -> list[dict]: ...


_WRITE_RE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+apoc\.(?!meta))\b", re.I)


def assert_read_only(q: str) -> None:
    if _WRITE_RE.search(q):
        raise ValueError("Write clauses are not allowed in generated Cypher")


# ======================================================================= Neo4j
class Neo4jGraph:
    def __init__(self, uri=None, user=None, password=None):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            uri or settings.neo4j_uri, auth=(user or settings.neo4j_user, password or settings.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def _run(self, q: str, **params):
        with self.driver.session() as s:
            return list(s.run(q, **params))

    # ---------- schema / writes
    def ensure_schema(self) -> None:
        self._run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
        self._run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name_lc)")
        self._run(
            "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (n:Entity) ON n.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}",
            dim=settings.embed_dim,
        )

    def upsert_nodes(self, label: str, rows: list[dict]) -> None:
        assert label in LABELS, label
        q = (
            f"UNWIND $rows AS row MERGE (n:Entity {{id: row.id}}) SET n:{label} "
            "SET n += row.props, n.name = row.name, n.name_lc = toLower(row.name), n.text = row.text "
            "WITH n, row WHERE row.embedding IS NOT NULL SET n.embedding = row.embedding"
        )
        for i in range(0, len(rows), 500):
            self._run(q, rows=rows[i : i + 500])

    def upsert_edges(self, rel: str, rows: list[dict]) -> None:
        assert rel in REL_TYPES, rel
        q = (
            "UNWIND $rows AS row MATCH (a:Entity {id: row.src}) MATCH (b:Entity {id: row.dst}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r += coalesce(row.props, {{}})"
        )
        for i in range(0, len(rows), 1000):
            self._run(q, rows=rows[i : i + 1000])

    def finalize(self) -> None:
        self._run("MATCH (n:Entity) SET n.degree = COUNT { (n)--() }")
        self._run("CALL db.awaitIndexes(300)")

    # ---------- reads used by the walker
    def vector_search(self, vec, k, labels=None):
        recs = self._run(
            "CALL db.index.vector.queryNodes('entity_embedding', $k, $vec) YIELD node, score "
            "RETURN node.id AS id, node.name AS name, "
            "[l IN labels(node) WHERE l <> 'Entity'][0] AS label, score",
            k=k * (3 if labels else 1), vec=list(vec),
        )
        out = [({"id": r["id"], "name": r["name"], "label": r["label"]}, float(r["score"])) for r in recs]
        if labels:
            out = [o for o in out if o[0]["label"] in labels]
        return out[:k]

    def find_by_name(self, names, exact_only=False):
        """Exact (case-insensitive) name matches; prefix matches only if nothing matches exactly."""
        names = [n.strip() for n in names if n and n.strip()]
        if not names:
            return []
        ret = "RETURN DISTINCT n.id AS id, n.name AS name, [l IN labels(n) WHERE l <> 'Entity'][0] AS label LIMIT 50"
        recs = self._run("UNWIND $names AS nm MATCH (n:Entity) WHERE n.name_lc = toLower(nm) " + ret, names=names)
        if not recs and not exact_only:
            long_names = [n for n in names if len(n) >= 4]
            recs = self._run("UNWIND $names AS nm MATCH (n:Entity) WHERE n.name_lc STARTS WITH toLower(nm) "
                             + ret.replace("LIMIT 50", "LIMIT 20"), names=long_names) if long_names else []
        return [dict(r) for r in recs]

    def common_neighbors(self, ids, label, limit=50):
        """Nodes with `label` adjacent to ALL of `ids` (exact set intersection, e.g. batch AND sector)."""
        if not ids:
            return []
        recs = self._run(
            f"MATCH (a:Entity {{id: $first}})--(m:{label}) "
            "WHERE ALL(x IN $rest WHERE EXISTS { MATCH (m)--(:Entity {id: x}) }) "
            "RETURN DISTINCT m.id AS id, m.name AS name LIMIT $limit",
            first=ids[0], rest=list(ids[1:]), limit=limit,
        ) if label in LABELS else []
        return [dict(r) for r in recs]

    def neighbors(self, ids, cap=64):
        recs = self._run(
            """
            UNWIND $ids AS nid
            MATCH (n:Entity {id: nid})
            CALL {
              WITH n
              MATCH (n)-[r]-(m:Entity)
              WITH r, m, startNode(r) = n AS outgoing ORDER BY rand() LIMIT $cap
              RETURN type(r) AS rel, outgoing, m
            }
            RETURN nid, rel, outgoing, m.id AS mid, m.name AS name,
                   [l IN labels(m) WHERE l <> 'Entity'][0] AS label, coalesce(m.degree, 1) AS degree
            """,
            ids=list(ids), cap=cap,
        )
        out: dict[str, list[Nbr]] = defaultdict(list)
        for r in recs:
            out[r["nid"]].append(Nbr(r["rel"], r["outgoing"], r["mid"], r["label"], r["name"], r["degree"]))
        return out

    def get_embeddings(self, ids):
        recs = self._run(
            "UNWIND $ids AS nid MATCH (n:Entity {id: nid}) WHERE n.embedding IS NOT NULL RETURN n.id AS id, n.embedding AS e",
            ids=list(ids),
        )
        return {r["id"]: np.asarray(r["e"], dtype=np.float32) for r in recs}

    def get_nodes(self, ids):
        recs = self._run(
            "UNWIND $ids AS nid MATCH (n:Entity {id: nid}) "
            "RETURN n.id AS id, [l IN labels(n) WHERE l <> 'Entity'][0] AS label, "
            "properties(n) AS props",
            ids=list(ids),
        )
        out = {}
        for r in recs:
            p = dict(r["props"])
            p.pop("embedding", None)
            p.pop("name_lc", None)
            out[r["id"]] = {"id": r["id"], "label": r["label"], **p}
        return out

    def edges_among(self, ids):
        recs = self._run(
            "MATCH (a:Entity)-[r]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids "
            "RETURN a.id AS s, type(r) AS t, b.id AS d",
            ids=list(ids),
        )
        return [(r["s"], r["t"], r["d"]) for r in recs]

    def schema_summary(self) -> str:
        labels = self._run(
            "MATCH (n:Entity) WITH [l IN labels(n) WHERE l <> 'Entity'][0] AS l RETURN l, count(*) AS c"
        )
        rels = self._run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WITH [l IN labels(a) WHERE l <> 'Entity'][0] AS s, type(r) AS t, [l IN labels(b) WHERE l <> 'Entity'][0] AS d "
            "RETURN s, t, d, count(*) AS c"
        )
        props = self._run(
            "MATCH (n:Entity) WITH [l IN labels(n) WHERE l <> 'Entity'][0] AS l, keys(n) AS ks "
            "UNWIND ks AS k WITH l, k WHERE NOT k IN ['embedding','text','name_lc'] "
            "RETURN l, collect(DISTINCT k) AS keys"
        )
        return _format_schema(
            {r["l"]: r["c"] for r in labels},
            [(r["s"], r["t"], r["d"], r["c"]) for r in rels],
            {r["l"]: r["keys"] for r in props},
        )

    def run_cypher(self, query, params=None, limit=50):
        assert_read_only(query)
        with self.driver.session() as s:
            res = s.execute_read(lambda tx: list(tx.run(query, **(params or {}))))
        return [r.data() for r in res[:limit]]


def _format_schema(label_counts, rel_triples, props) -> str:
    lines = ["Every node also has label :Entity and properties id, name. Node labels:"]
    for l, c in sorted(label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  (:{l}) count={c} props={sorted(props.get(l, []))}")
    lines.append("Relationships:")
    for s, t, d, c in sorted(rel_triples, key=lambda x: -x[3]):
        lines.append(f"  (:{s})-[:{t}]->(:{d}) count={c}")
    return "\n".join(lines)


# ======================================================================= In-memory
class InMemoryGraph:
    """Same interface as Neo4jGraph; adequate for a few 10k nodes."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.emb: dict[str, np.ndarray] = {}
        self.adj: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)  # id -> (rel, outgoing, other)
        self.edges: set[tuple[str, str, str]] = set()
        self._mat = None
        self._mat_ids: list[str] = []

    def ensure_schema(self):
        pass

    def upsert_nodes(self, label, rows):
        for r in rows:
            node = self.nodes.setdefault(r["id"], {"id": r["id"]})
            node.update(r.get("props", {}))
            node.update({"label": label, "name": r["name"], "text": r.get("text", "")})
            if r.get("embedding") is not None:
                self.emb[r["id"]] = np.asarray(r["embedding"], dtype=np.float32)
        self._mat = None

    def upsert_edges(self, rel, rows):
        for r in rows:
            if r["src"] not in self.nodes or r["dst"] not in self.nodes:
                continue
            key = (r["src"], rel, r["dst"])
            if key in self.edges:
                continue
            self.edges.add(key)
            self.adj[r["src"]].append((rel, True, r["dst"]))
            self.adj[r["dst"]].append((rel, False, r["src"]))

    def finalize(self):
        for nid, n in self.nodes.items():
            n["degree"] = len(self.adj.get(nid, []))

    def _matrix(self):
        if self._mat is None:
            self._mat_ids = list(self.emb)
            m = np.stack([self.emb[i] for i in self._mat_ids]) if self._mat_ids else np.zeros((0, settings.embed_dim))
            self._mat = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        return self._mat

    def vector_search(self, vec, k, labels=None):
        m = self._matrix()
        if not len(m):
            return []
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        scores = m @ v
        out = []
        for idx in np.argsort(-scores):
            n = self.nodes[self._mat_ids[idx]]
            if labels and n["label"] not in labels:
                continue
            out.append(({"id": n["id"], "name": n["name"], "label": n["label"]}, float(scores[idx])))
            if len(out) >= k:
                break
        return out

    def find_by_name(self, names, exact_only=False):
        names_lc = [n.strip().lower() for n in names if n and n.strip()]
        hit = lambda pred: [{"id": n["id"], "name": n["name"], "label": n["label"]}
                            for n in self.nodes.values() if any(pred(n["name"].lower(), q) for q in names_lc)]
        out = hit(lambda name, q: name == q)
        if not out and not exact_only:
            out = hit(lambda name, q: len(q) >= 4 and name.startswith(q))[:20]
        return out[:50]

    def common_neighbors(self, ids, label, limit=50):
        if not ids:
            return []
        sets = [{m for _, _, m in self.adj.get(i, [])} for i in ids]
        common = set.intersection(*sets)
        return [{"id": m, "name": self.nodes[m]["name"]} for m in sorted(common)
                if self.nodes[m]["label"] == label][:limit]

    def neighbors(self, ids, cap=64, rng=np.random.default_rng(0)):
        out = {}
        for nid in ids:
            lst = self.adj.get(nid, [])
            if len(lst) > cap:
                lst = [lst[i] for i in rng.choice(len(lst), cap, replace=False)]
            out[nid] = [
                Nbr(rel, o, m, self.nodes[m]["label"], self.nodes[m]["name"], self.nodes[m].get("degree", 1))
                for rel, o, m in lst
            ]
        return out

    def get_embeddings(self, ids):
        return {i: self.emb[i] for i in ids if i in self.emb}

    def get_nodes(self, ids):
        return {i: {k: v for k, v in self.nodes[i].items()} for i in ids if i in self.nodes}

    def edges_among(self, ids):
        s = set(ids)
        return [e for e in self.edges if e[0] in s and e[2] in s]

    def schema_summary(self):
        lc, props, trip = defaultdict(int), defaultdict(set), defaultdict(int)
        for n in self.nodes.values():
            lc[n["label"]] += 1
            props[n["label"]].update(k for k in n if k not in ("text", "label"))
        for s, t, d in self.edges:
            trip[(self.nodes[s]["label"], t, self.nodes[d]["label"])] += 1
        return _format_schema(lc, [(*k, c) for k, c in trip.items()], {k: list(v) for k, v in props.items()})

    def run_cypher(self, query, params=None, limit=50):
        raise NotImplementedError("Cypher execution requires the Neo4j backend")

    # persistence for tests
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"nodes": self.nodes, "emb": {k: v.tolist() for k, v in self.emb.items()},
                       "edges": sorted(self.edges)}, f)

    @classmethod
    def load(cls, path):
        g = cls()
        with open(path) as f:
            d = json.load(f)
        g.nodes = d["nodes"]
        g.emb = {k: np.asarray(v, dtype=np.float32) for k, v in d["emb"].items()}
        for s, t, dd in d["edges"]:
            g.upsert_edges(t, [{"src": s, "dst": dd}])
        g.finalize()
        return g


def get_graph(backend: str = "neo4j") -> GraphBackend:
    return Neo4jGraph() if backend == "neo4j" else InMemoryGraph()
