"""Collection-layer evaluation: investor edges extracted from public news vs PitchBook (golden set).

PitchBook is used ONLY here, never loaded into the graph the agent queries.

    python scripts/eval_collection.py                       # reads data/pitchbook/*PitchBook*.xlsx
    python scripts/eval_collection.py --pb-dir some/folder

Outputs (results/collection_eval/):
    summary.json   precision / recall / F1 of (company, investor) edges, lead precision, round coverage
    edges.csv      every TP / FP / FN edge with its source URL -> audit the FPs by hand
                   (PitchBook is not complete either: an FP can be a real investor PitchBook lacks)
    rounds.csv     PitchBook financing rounds and whether an extracted event matched them
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from vcgraph.collect.investors import investor_key, normalize_round  # noqa: E402
from vcgraph.collect.resolve import domain_of, normalize_name  # noqa: E402

INVESTMENT_TYPES = {"Accelerator/Incubator", "Seed Round", "Early Stage VC", "Later Stage VC", "Angel (individual)",
                    "Angel (group)", "Corporate", "Equity Crowdfunding", "Convertible Debt"}
EXCLUDE_TYPE2 = {"Non-Equity Accelerator"}
YC_KEY = investor_key("Y Combinator")


# ------------------------------------------------------------------ PitchBook gold
def load_pitchbook(pb_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(pb_dir, "*PitchBook*.xlsx"))):
        raw = pd.read_excel(f, header=None)
        hdr = next(i for i, r in raw.iterrows() if "Deal ID" in [str(v).strip() for v in r])
        df = pd.read_excel(f, header=hdr).copy()
        df["__file"] = os.path.basename(f)
        frames.append(df)
    if not frames:
        sys.exit(f"no *PitchBook*.xlsx files in {pb_dir}")
    d = pd.concat(frames, ignore_index=True).dropna(subset=["Deal ID"])
    d = d[~d["Deal ID"].astype(str).str.contains("PitchBook|©|Total", regex=True)]
    return d.drop_duplicates("Deal ID")


def split_investors(cell) -> list[str]:
    """'A, B(Partner X, Partner Y), C (Utah)' -> ['A', 'B', 'C (Utah)']: commas inside () are not separators."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    out, depth, cur = [], 0, ""
    for ch in cell:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    # "Initialized Capital(Zoe Perret)" -> partner name in parentheses glued to the firm: drop it
    return [re.sub(r"(?<=\S)\(.*$", "", x).strip() for x in out if x.strip()]


def gold_edges(pb: pd.DataFrame) -> tuple[dict, dict, list[dict]]:
    """company PBId -> {investor_key: display}, company PBId -> {lead keys}, list of rounds."""
    inv, leads, rounds = defaultdict(dict), defaultdict(set), []
    for _, r in pb.iterrows():
        if r.get("Deal Type") not in INVESTMENT_TYPES or r.get("Deal Type 2") in EXCLUDE_TYPE2:
            continue
        cid = str(r["Company ID"])
        for name in split_investors(r.get("Investors")):
            inv[cid][investor_key(name)] = name
        for name in split_investors(r.get("Lead/Sole Investors")):
            leads[cid].add(investor_key(name))
            inv[cid].setdefault(investor_key(name), name)
        if r.get("Deal Type") != "Accelerator/Incubator":
            size = r.get("Deal Size")
            rounds.append({"pb_company_id": cid, "company": r["Companies"], "deal_id": r["Deal ID"],
                           "date": str(r.get("Deal Date"))[:10], "type": r.get("Deal Type"),
                           "round": normalize_round(str(r.get("Deal Type 2") or r.get("Deal Type"))),
                           "amount_usd": float(size) * 1e6 if pd.notna(size) else None})
    return inv, leads, rounds


# ------------------------------------------------------------------ mapping PitchBook -> our companies
def map_companies(pb: pd.DataFrame, entities: list[dict]) -> dict[str, dict]:
    by_dom, by_name = {}, {}
    for e in entities:
        a = e.get("attrs") or {}
        d = domain_of(a.get("website"))
        if d:
            by_dom.setdefault(d, e)
        by_name.setdefault(normalize_name(e["name"]), e)
    out = {}
    for _, r in pb.drop_duplicates("Company ID").iterrows():
        cid = str(r["Company ID"])
        e = by_dom.get(domain_of(r.get("Company Website"))) or by_name.get(normalize_name(str(r["Companies"])))
        if e:
            out[cid] = e
    return out


# ------------------------------------------------------------------ evaluation
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    rcl = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * rcl / (p + rcl) if p and rcl else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": _r(p), "recall": _r(rcl), "f1": _r(f1)}


def _r(x):
    return round(x, 3) if x is not None else None


def evaluate(pb: pd.DataFrame, entities: list[dict], news: list[dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    gold, gold_leads, gold_rounds = gold_edges(pb)
    mapping = map_companies(pb, entities)
    via_of_url = {r["data"]["url"]: r["data"].get("via") for r in news}
    events_by_ent = defaultdict(list)
    for r in news:
        events_by_ent[r["data"]["entity_id"]].append(r["data"])

    rows = []
    for cid, ent in mapping.items():
        pred = {}
        for i in ent.get("investors", []):
            if isinstance(i, dict):
                pred[investor_key(i["name"])] = i
            else:
                pred[investor_key(i)] = {"name": i}
        g = gold.get(cid, {})
        for k in set(g) | set(pred):
            status = "TP" if k in g and k in pred else "FN" if k in g else "FP"
            p = pred.get(k, {})
            rows.append({"company": ent["name"], "entity_id": ent["entity_id"], "investor_key": k,
                         "gold_name": g.get(k), "pred_name": p.get("name"), "status": status,
                         "is_yc": k == YC_KEY, "gold_lead": k in gold_leads.get(cid, set()),
                         "pred_lead": bool(p.get("lead")), "source_url": p.get("source_url"),
                         "via": via_of_url.get(p.get("source_url")), "company_has_news": bool(events_by_ent.get(ent["entity_id"]))})
    edges = pd.DataFrame(rows)

    def score(df):
        return prf(int((df.status == "TP").sum()), int((df.status == "FP").sum()), int((df.status == "FN").sum()))

    summary = {
        "pitchbook_companies": int(pb["Company ID"].nunique()),
        "mapped_to_graph": len(mapping),
        "pitchbook_investment_deals": int(sum(1 for _, r in pb.iterrows() if r.get("Deal Type") in INVESTMENT_TYPES)),
        "gold_investor_edges": int(sum(len(v) for k, v in gold.items() if k in mapping)),
        "companies_with_extracted_news": int(sum(1 for e in mapping.values() if events_by_ent.get(e["entity_id"]))),
    }
    if not edges.empty:
        summary["edges_all"] = score(edges)
        summary["edges_excluding_yc"] = score(edges[~edges.is_yc])
        summary["edges_companies_with_news"] = score(edges[edges.company_has_news])
        summary["edges_companies_with_news_excl_yc"] = score(edges[edges.company_has_news & ~edges.is_yc])
        pl = edges[edges.pred_lead]
        summary["lead_precision"] = _r((pl.gold_lead & (pl.status == "TP")).mean()) if len(pl) else None
        gl = edges[edges.gold_lead]
        summary["lead_recall"] = _r((gl.pred_lead).mean()) if len(gl) else None
        by_via = {}
        for via, g in edges[edges.status != "FN"].groupby(edges["via"].fillna("other")):
            by_via[via] = {"pred_edges": len(g), "precision": _r((g.status == "TP").mean())}
        summary["precision_by_source"] = by_via

    # rounds: does any extracted event match a PitchBook round (amount within 15%, or same round name)?
    rr = []
    for g in gold_rounds:
        ent = mapping.get(g["pb_company_id"])
        evs = events_by_ent.get(ent["entity_id"], []) if ent else []
        hit = None
        for ev in evs:
            a = ev.get("amount_usd")
            if g["amount_usd"] and a and abs(a - g["amount_usd"]) <= 0.15 * g["amount_usd"]:
                hit = ev
                break
            if g["round"] and normalize_round(ev.get("round")) == g["round"]:
                hit = ev
        rr.append({**g, "mapped": bool(ent), "matched": bool(hit),
                   "extracted_amount": hit.get("amount_usd") if hit else None,
                   "extracted_round": hit.get("round") if hit else None,
                   "source_url": hit.get("url") if hit else None})
    rounds = pd.DataFrame(rr)
    if not rounds.empty:
        m = rounds[rounds.mapped]
        summary["rounds"] = {"pitchbook_rounds": len(m), "matched": int(m.matched.sum()),
                             "round_recall": _r(m.matched.mean()) if len(m) else None}
        with_amt = m[m.matched & m.amount_usd.notna() & m.extracted_amount.notna()]
        if len(with_amt):
            err = (with_amt.extracted_amount - with_amt.amount_usd).abs() / with_amt.amount_usd
            summary["rounds"]["amount_within_15pct"] = _r((err <= 0.15).mean())
    return summary, edges, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb-dir", default="data/pitchbook")
    args = ap.parse_args()
    from vcgraph.storage import mongo

    pb = load_pitchbook(args.pb_dir)
    summary, edges, rounds = evaluate(pb, mongo.load_entities(), mongo.load_records("news"))
    out = ROOT / "results" / "collection_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    edges.to_csv(out / "edges.csv", index=False)
    rounds.to_csv(out / "rounds.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}/summary.json, edges.csv, rounds.csv")


if __name__ == "__main__":
    main()