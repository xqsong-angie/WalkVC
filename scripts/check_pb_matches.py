"""Check PitchBook 'Upload a List' match quality and save reliable YC -> PitchBook matches.

Usage:
    python scripts/check_pb_matches.py data/pitchbook/PitchBook_Upload_List_Results_2026_10_28.csv

Output:
    data/pitchbook/yc_pb_matched.csv   reliable matches (Company Name, Website, YC Batch, PBId, ...)
    data/pitchbook/yc_pb_review.csv    matches that need a manual look
"""
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

SUFFIX = r"\s+(incorporated|inc|corp|corporation|co|company|llc|ltd|limited|plc|gmbh|labs?|ai|hq|technologies)$"


def domain(s):
    """'https://www.foo.com/x' or '[www.foo.com](https://www.foo.com)' -> 'foo.com'; anything empty -> None."""
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    m = re.search(r"\]\(([^)]+)\)", s)          # markdown link -> take the URL part
    if m:
        s = m.group(1)
    s = s.strip("[]")
    if "://" not in s:
        s = "http://" + s
    host = urllib.parse.urlparse(s).netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None


def root(d):
    """app.foo.com -> foo.com (keeps foo.co.uk style domains mostly intact)."""
    if not isinstance(d, str):
        return None
    parts = d.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def norm_name(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\(.*?\)", " ", s.lower())       # "Onyx (Business/Productivity Software)" -> "onyx"
    s = re.sub(r"[^a-z0-9 ]+", " ", s).strip()
    for _ in range(2):
        s = re.sub(SUFFIX, "", s).strip()
    return re.sub(r"\s+", " ", s)


def main(path):
    df = pd.read_csv(path)
    df["yc_dom"] = df["Website"].map(domain)
    df["pb_dom"] = df["Website from PitchBook"].map(domain)
    df["name_sim"] = [fuzz.token_sort_ratio(norm_name(a), norm_name(b))
                      for a, b in zip(df["Company Name"], df["Firm Name from PitchBook"])]

    has_match = df["PBId"].notna()
    same_domain = df["yc_dom"].map(root).notna() & (df["yc_dom"].map(root) == df["pb_dom"].map(root))
    strong = df["Score"].isin(["Very High", "High"]) & (df["name_sim"] >= 90)

    df["status"] = "unmatched"
    df.loc[has_match, "status"] = "check"
    df.loc[has_match & (same_domain | strong), "status"] = "ok"
    df["reason"] = ""
    df.loc[has_match & same_domain, "reason"] = "same_domain"
    df.loc[has_match & ~same_domain & strong, "reason"] = "high_score+name"

    print("=== status ===")
    print(df["status"].value_counts().to_string())
    print("\n=== Score x status ===")
    print(pd.crosstab(df["Score"].fillna("(blank)"), df["status"]).to_string())

    cols = ["Company Name", "Website", "Firm Name from PitchBook", "Website from PitchBook", "Score", "name_sim"]
    review = df[df["status"] == "check"]
    print(f"\n=== needs manual review: {len(review)} (first 40) ===")
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(review[cols].head(40).to_string())

    out = Path(path).parent
    df[df["status"] == "ok"][["Company Name", "Website", "YC Batch", "PBId", "Firm Name from PitchBook",
                              "Score", "reason"]].to_csv(out / "yc_pb_matched.csv", index=False)
    review[cols + ["YC Batch", "PBId"]].to_csv(out / "yc_pb_review.csv", index=False)
    print(f"\nsaved {(df['status'] == 'ok').sum()} reliable matches -> {out / 'yc_pb_matched.csv'}")
    print(f"saved {len(review)} rows to review       -> {out / 'yc_pb_review.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "data/pitchbook/PitchBook_Upload_List_Results_2026_10_28.csv")