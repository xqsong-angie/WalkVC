"""Why is investor recall low? Source coverage vs extraction.

For every PitchBook company that has accepted funding news, take its PitchBook investors (excluding
Y Combinator) and check whether each one is mentioned ANYWHERE in the fetched article text.
  * mentioned in text  -> the extractor could have found it (ceiling for the LLM)
  * extracted          -> the LLM actually did
So:  recall = P(mentioned in text) x P(extracted | mentioned)
             = source coverage      x extraction recall

    python scripts/collection_ceiling.py              # reads data/pitchbook/*PitchBook*.xlsx
    python scripts/collection_ceiling.py --pb-dir some/folder

Writes results/collection_eval/ceiling.csv (one row per PitchBook investor edge) and prints a summary.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import eval_collection as ev  # noqa: E402
from vcgraph.collect.investors import investor_key  # noqa: E402


def mentioned(name: str, key: str, text: str) -> bool:
    core = name.lower().split(" (")[0].strip()
    return bool(core and core in text) or (len(key) >= 4 and key in text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb-dir", default="data/pitchbook")
    args = ap.parse_args()
    from vcgraph.storage import mongo

    pb = ev.load_pitchbook(str(ROOT / args.pb_dir))
    gold, gold_leads, _ = ev.gold_edges(pb)
    mapping = ev.map_companies(pb, mongo.load_entities())
    urls = defaultdict(list)
    for r in mongo.load_records("news"):
        urls[r["data"]["entity_id"]].append(r["data"]["url"])

    rows = []
    for cid, ent in mapping.items():
        eid = ent["entity_id"]
        if eid not in urls:
            continue
        pages = [(mongo.get_raw(u) or {}).get("meta", {}).get("markdown", "") for u in urls[eid]]
        text = " ".join(pages).lower()
        pred = {investor_key(i["name"] if isinstance(i, dict) else i) for i in ent.get("investors", [])}
        for k, name in gold.get(cid, {}).items():
            if k == ev.YC_KEY:
                continue
            m = mentioned(name, k, text)
            rows.append({"company": ent["name"], "investor": name, "investor_key": k,
                         "gold_lead": k in gold_leads.get(cid, set()),
                         "n_articles": len(pages), "article_chars": len(text),
                         "mentioned_in_text": m, "extracted": k in pred})

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "collection_eval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ceiling.csv", index=False)
    if df.empty:
        sys.exit("no PitchBook company with accepted funding news -> run `python main.py investors` first")

    def block(d):
        tot, men = len(d), int(d.mentioned_in_text.sum())
        ext_m = int((d.mentioned_in_text & d.extracted).sum())
        return {"gold_edges": tot,
                "mentioned_in_text": men,
                "source_coverage": round(men / tot, 3) if tot else None,
                "extracted_given_mentioned": ext_m,
                "extraction_recall": round(ext_m / men, 3) if men else None,
                "extracted_total": int(d.extracted.sum()),
                "end_to_end_recall": round(d.extracted.mean(), 3) if tot else None}

    summary = {"companies_with_news": int(df.company.nunique()),
               "all_investors": block(df),
               "lead_investors_only": block(df[df.gold_lead])}
    (out / "ceiling.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nper company:")
    print(df.groupby("company").agg(gold=("investor", "size"), in_text=("mentioned_in_text", "sum"),
                                    extracted=("extracted", "sum"), articles=("n_articles", "first"))
          .sort_values("gold", ascending=False).to_string())
    print(f"\nsaved -> {out}/ceiling.csv, ceiling.json")


if __name__ == "__main__":
    main()