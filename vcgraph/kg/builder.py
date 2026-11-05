"""Resolved entities (Mongo `entities`) -> Neo4j knowledge graph with vector embeddings.

Schema
  (:Person)-[:FOUNDED]->(:Company)          (:Investor)-[:INVESTED_IN {round, amount}]->(:Company)
  (:Person)-[:WORKED_AT {role}]->(:Company)  (:Company)-[:IN_SECTOR]->(:Sector)
  (:Company)-[:IN_BATCH]->(:Batch)           (:Company)-[:FILED]->(:Filing)   (SEC Form D)
  (:Person)-[:RELATED_TO_FILING {roles}]->(:Filing)
  (:Company)-[:REGISTERED_AS]->(:LegalEntity)  (OpenCorporates)
  (:Company)-[:SIMILAR_TO {score}]->(:Company)  (derived: embedding kNN, helps competitor discovery)
All nodes also carry :Entity {id, name, text, embedding, degree}.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..collect.investors import investor_node_id
from ..collect.resolve import slugify
from ..llm import embed
from ..storage.graph import GraphBackend


def person_id(name: str, company: str) -> str:
    """Full names are shared across companies (founder networks). A bare first name ("Will", "Alex")
    is too ambiguous to link companies, so it gets a company-scoped id."""
    name = name.strip()
    pid = f"person:{slugify(name)}"
    return pid if len(name.split()) > 1 else f"{pid}--{slugify(company)}"


def _company_text(e: dict) -> str:
    a = e.get("attrs", {})
    return " ".join(filter(None, [
        f"Company: {e['name']}.", a.get("one_liner"), (a.get("description") or "")[:600],
        f"Sectors: {', '.join(e.get('sectors', []))}." if e.get("sectors") else None,
        f"Tags: {', '.join(a.get('tags') or [])}." if a.get("tags") else None,
        f"Batch {a['batch']}." if a.get("batch") else None,
        f"Location {a['location']}." if a.get("location") else None,
    ]))


def entities_to_graph_rows(entities: list[dict]):
    nodes: dict[str, dict[str, dict]] = defaultdict(dict)   # label -> id -> row
    edges: dict[str, list[dict]] = defaultdict(list)

    def node(label, nid, name, text, **props):
        cur = nodes[label].get(nid)
        if cur is None:
            nodes[label][nid] = {"id": nid, "name": name, "text": text,
                                 "props": {k: v for k, v in props.items() if v is not None}}
        else:
            cur["props"].update({k: v for k, v in props.items() if v is not None})
        return nid

    for e in entities:
        a = e.get("attrs", {})
        filings = e.get("filings", [])
        raised = sum(f.get("total_amount_sold") or 0 for f in filings) or a.get("total_raised")
        last_raise = max((f.get("date_first_sale") or f.get("file_date") or "" for f in filings), default=None) \
            or a.get("last_raise_date")
        cid = node("Company", e["entity_id"], e["name"], _company_text(e),
                   one_liner=a.get("one_liner"), description=(a.get("description") or "")[:1000] or None,
                   website=a.get("website"), batch=a.get("batch"), location=a.get("location"),
                   team_size=a.get("team_size"), status=a.get("status"), founded_year=a.get("founded_year"),
                   total_raised=raised or None, last_raise_date=last_raise or None,
                   industry=a.get("industry"), subindustry=a.get("subindustry"), stage=a.get("stage"),
                   tags=a.get("tags") or None, yc_url=a.get("yc_url"),
                   sources=e.get("sources"), aliases=e.get("aliases") or None)

        for s in e.get("sectors", []):
            sid = node("Sector", f"sector:{slugify(s)}", s, f"Sector: {s}")
            edges["IN_SECTOR"].append({"src": cid, "dst": sid})
        if a.get("batch"):
            bid = node("Batch", f"batch:{slugify(a['batch'])}", a["batch"], f"Y Combinator batch {a['batch']}")
            edges["IN_BATCH"].append({"src": cid, "dst": bid})

        for f in e.get("founders", []):
            pid = node("Person", person_id(f["name"], e["name"]), f["name"],
                       f"Person: {f['name']}, {f.get('title') or 'founder'} of {e['name']}.",
                       linkedin=f.get("linkedin"))
            edges["FOUNDED"].append({"src": pid, "dst": cid, "props": {"title": f.get("title")}})
            for prev in f.get("previously", []) or []:
                edges["WORKED_AT"].append({"src": pid, "dst": prev["company_id"],
                                           "props": {"role": prev.get("role")}})

        for inv in e.get("investors", []):
            inv_name = inv["name"] if isinstance(inv, dict) else inv
            iid = node("Investor", investor_node_id(inv_name), inv_name, f"Investor: {inv_name}.")
            props = ({k: inv.get(k) for k in ("round", "amount", "date", "lead", "n_rounds", "source_url")}
                     if isinstance(inv, dict) else {})
            edges["INVESTED_IN"].append({"src": iid, "dst": cid, "props": props})

        for f in filings:
            fid = node("Filing", f"filing:{f['accession']}", f"Form D {f['accession']}",
                       f"SEC Form D filing by {e['name']}: industry {f.get('industry_group')}, "
                       f"amount sold {f.get('total_amount_sold')}, first sale {f.get('date_first_sale')}.",
                       form="D", industry_group=f.get("industry_group"), file_date=f.get("file_date"),
                       date_first_sale=f.get("date_first_sale"), total_amount_sold=f.get("total_amount_sold"),
                       total_offering_amount=f.get("total_offering_amount"), state=f.get("state"),
                       num_investors=f.get("num_investors"), url=f.get("url"))
            edges["FILED"].append({"src": cid, "dst": fid})
            for p in f.get("related_persons", []):
                pid = node("Person", person_id(p["name"], e["name"]), p["name"],
                           f"Person: {p['name']}, {', '.join(p.get('roles', []))} at {e['name']}.")
                edges["RELATED_TO_FILING"].append({"src": pid, "dst": fid, "props": {"roles": p.get("roles")}})
                if any("Executive" in r for r in p.get("roles", [])):
                    edges["WORKED_AT"].append({"src": pid, "dst": cid, "props": {"role": "Executive Officer"}})

        for le in e.get("legal_entities", []):
            lid = node("LegalEntity", f"legal:{le['id']}", le.get("name") or e["name"],
                       f"Legal entity {le.get('name')} registered in {le.get('jurisdiction_code')}.",
                       jurisdiction=le.get("jurisdiction_code"), company_number=le.get("company_number"),
                       incorporation_date=le.get("incorporation_date"), status=le.get("current_status"),
                       url=le.get("opencorporates_url"))
            edges["REGISTERED_AS"].append({"src": cid, "dst": lid})

    return nodes, edges


def similar_edges(ids: list[str], vecs: np.ndarray, k: int = 3, threshold: float = 0.6) -> list[dict]:
    if len(ids) < 2:
        return []
    m = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = m @ m.T
    np.fill_diagonal(sims, -1)
    seen: set[tuple[int, int]] = set()
    rows = []
    for i in range(len(ids)):
        for j in np.argsort(-sims[i])[:k]:
            pair = (min(i, int(j)), max(i, int(j)))
            if sims[i, j] >= threshold and pair not in seen:
                seen.add(pair)
                rows.append({"src": ids[pair[0]], "dst": ids[pair[1]],
                             "props": {"score": round(float(sims[i, j]), 4)}})
    return rows


def build_graph(entities: list[dict], g: GraphBackend, sim_k: int = 3, sim_threshold: float = 0.6,
                verbose: bool = True) -> dict:
    g.ensure_schema()
    nodes, edges = entities_to_graph_rows(entities)
    stats = {"nodes": {}, "edges": {}}
    company_vecs = None
    for label, rows_by_id in nodes.items():
        rows = list(rows_by_id.values())
        vecs = embed([r["text"] for r in rows])
        for r, v in zip(rows, vecs):
            r["embedding"] = v
        g.upsert_nodes(label, rows)
        stats["nodes"][label] = len(rows)
        if label == "Company":
            company_vecs = ([r["id"] for r in rows], np.asarray(vecs, dtype=np.float32))
        if verbose:
            print(f"  upserted {len(rows):5d} :{label}")
    if company_vecs:
        edges["SIMILAR_TO"] = similar_edges(*company_vecs, k=sim_k, threshold=sim_threshold)
    for rel, rows in edges.items():
        g.upsert_edges(rel, rows)
        stats["edges"][rel] = len(rows)
        if verbose:
            print(f"  upserted {len(rows):5d} -[:{rel}]->")
    g.finalize()
    return stats
