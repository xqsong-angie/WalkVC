# Results index

Which folder holds which experiment, on which graph version, and where it is used in the report.
Nothing here is ever overwritten: reruns get new folder names.

## Graph versions

| Version | Built | Contents | Notes |
|---|---|---|---|
| **G0** | Oct 30 | 72 synthetic companies (`python main.py demo`) | development only |
| **G1** | Nov 2 | 1,829 YC companies, founders, 57 SEC Form D filings, 17 investor edges | after entity-resolution fixes (no YC-YC merges, implausible Form D removed, first-name-only founders not merged) |
| **G2** | Nov 6 | G1 + news-extracted investors: ~546 investors, 927 `INVESTED_IN` edges | built by `python main.py investors --rss` |
| **G2-final** | Nov 7 | G2 minus ~13 stale Investor nodes from the old id scheme | `scripts/final_rerun.py`; every main-table number comes from G2 / G2-final (the stale nodes touch < 2% of investor edges) |

Question sets: `data/yc/questions.jsonl` (46 point queries), `data/yc/questions_explore.jsonl` (12 open-ended + 12 multi-hop),
`data/yc/questions_coinvest.jsonl` (20 co-investment + 20 shared-investor, two-hop). Demo questions: `data/demo/questions.jsonl`.

## Main results (report body) — G2 / G2-final

| Folder | What | Report section |
|---|---|---|
| `point_final_cypher`, `point_final_anchor_only`, `point_final_walk12` | point queries: Text-to-Cypher vs anchoring vs anchoring + walk (12 nodes) | Reasoning layer: point queries |
| `explore_final_anchor_only`, `explore_final_v1` (walk 12 + expand 16k), `explore_final_expand_2500`, `explore_final_expand_7000`, `explore_final_walk_40` | open-ended + founder multi-hop, token-budget comparison | Reasoning layer: exploration / budget |
| `explore_final_budget/` | tables, paired bootstrap, `recall_vs_tokens.png` for the set above | idem (figure) |
| `coinv_anchor_only`, `coinv_v1`, `coinv_expand_2500`, `coinv_expand_7000`, `coinv_walk_40` | co-investment / shared-investor two-hop questions | Reasoning layer: two-hop |
| `coinv_budget/` | tables, bootstrap, figure for the set above | idem (figure) |
| `explore_cypher`, `coinv_cypher` | Text-to-Cypher baseline on the two-hop / exploration sets | Baselines |
| `answer_recall_final/` | answer-level recall for every method on every set (`scripts/answer_recall.py --final`) | Main comparison table |
| `two_hop_summary_explore_final.csv` | all two-hop questions pooled + size-aware router simulation | Router / discussion |
| `explore_walk40_gain`, `explore_walk40_fixed`, `explore_walk40_eps15`, `coinv_walk40_fixed`, `coinv_walk40_eps15` (+ `coinv_walk_40`) | stopping-rule ablation | Ablation: information-gain stopping |
| `stopping_ablation/` | summary + bootstrap for the ablation | idem |

## Collection layer (independent of graph version: reads MongoDB)

| Folder | What | Report section |
|---|---|---|
| `collection_eval/summary.json`, `edges.csv`, `rounds.csv` | news-extracted investors vs PitchBook golden set (43 companies, 324 edges) | Collection layer evaluation |
| `collection_eval/ceiling.json`, `ceiling.csv` | recall decomposition: source coverage (7%) x extraction recall (75%) | idem |

PitchBook exports stay in `data/pitchbook/` (git-ignored, licensed data, never loaded into the graph).

## Development history (use only for "how we got here"; compare only within the same graph)

| Folder | Graph | What |
|---|---|---|
| `run-2026103*`, `fast`, `top12`, `nocheck`, `baseline` | G0 | first latency / ablation runs on synthetic data (top-12 nodes, sufficiency check) |
| `yc_v1` | G1 | first real-data run: walk only (no anchoring) + Cypher; `ragas_cypher.csv` = RAGAS of the Cypher baseline |
| `yc_v2` (+ `ragas.csv`), `yc_v2_anchor_only` | G1 | anchoring added (recall 0.55 -> 0.87); anchor-only ablation |
| `explore_v1`, `explore_anchor_only`, `explore_expand_2500`, `explore_expand_7000`, `explore_walk_40`, `explore_budget/` | G1 | first exploration / budget runs (superseded by `explore_final_*`, kept as a stability check) |
| `answer_recall/`, `two_hop_summary.csv` | G1 + G2 mixed | first answer-recall / pooled analysis (superseded by the `_final` versions) |

## Run-to-run noise

The stopping ablation compared two configurations that walk almost identically (`info_gain` vs `fixed_12`)
and still differed by about ±0.04-0.05 recall, mostly from the LLM decomposing the question differently.
Treat differences below ~0.05 that are not backed by a paired bootstrap CI as noise.