# Agentic KG Exploration via Random Walks for VC Market Intelligence

<a href="README_CN.md">
  <img src="https://img.shields.io/badge/Report-Full Version-FF85A1?style=for-the-badge" alt="Language">
</a>
<a href="DEPLOY.md">
  <img src="https://img.shields.io/badge/Deployment-Guide-C7CEEA?style=for-the-badge" alt="Topics">
</a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-70C1B3?style=for-the-badge" alt="License">
</a>

<img width="960" height="540" alt="architecture" src="https://github.com/user-attachments/assets/e0c457d9-bda5-45c0-8807-177be8a76f30" />


## 1. Quick start

```bash
cp .env.example .env
docker compose up -d                       # neo4j, mongo, ollama
docker exec vckg-ollama ollama pull llama3.1:8b
docker exec vckg-ollama ollama pull qwen2.5:14b
docker exec vckg-ollama ollama pull nomic-embed-text

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
crawl4ai-setup                             # one-time Playwright browser install (only for real crawling)

# 0) offline sanity check (hash embeddings, fake LLM, in-memory graph)
EMBED_BACKEND=hash pytest -q
```

### 1.1 Option A: synthetic demo graph

```bash
python main.py demo                         # 72 fictional companies -> Neo4j, writes data/demo/questions.jsonl
python main.py ask "Which startups compete with Junostack?" -v
python main.py ask "Who has co-invested alongside Northpeak Ventures?" --model qwen2.5:14b
python main.py ask "Which investors are most active in Fintech startups?" --mode cypher   # baseline
```

Note: Company names come from `data/demo/questions.jsonl`, you can choose company names from the file.

### 1.2 Option B: real Y Combinator data

```bash
# 1) collect: YC list (yc-oss JSON mirror) + crawl each www.ycombinator.com/companies/<slug> page for founders
#    -> SEC Form D enrichment -> OpenCorporates (if token) -> entity resolution -> Neo4j
python main.py collect --batches W25 S25 --limit 40            # --yc-mode hybrid is the default
#    --yc-mode json   : no browser, fastest, but NO founder names
#    --yc-mode crawl  : everything from www.ycombinator.com via Crawl4AI + LLM extraction (slowest)
#    --limit 0        : the whole batch
#    --skip-sec       : skip EDGAR if you only want YC

# 2) build an evaluation set from the real data (reference answers come from YC/SEC facts)
python main.py questions --out data/yc/questions.jsonl --per-type 8

# 3) evaluate on the real graph
python main.py ask "Which startups compete with <a W25 company>?" -v
python main.py eval --questions data/yc/questions.jsonl --models llama3.1:8b qwen2.5:14b gpt-4o
```

How ground truth is defined for real data (`vcgraph/eval/make_questions.py`), on purpose independent of
the walk and of the embedding-derived `SIMILAR_TO` edges:

| kind | question | reference |
|---|---|---|
| competitors | Which startups compete with X? | same YC sub-industry **and** ≥1 shared YC tag |
| founders | Who founded X? | founders listed on X's YC page |
| market_signal | What {industry} startups are in YC {batch}? | YC industry × batch |
| funding | How much has X raised per Form D? | SEC Form D `totalAmountSold` |
| network | Which companies is P connected to? | person appears at ≥2 companies (founder or Form D officer) |
| investors | Which investors backed X? | only if investors were extracted from pages |

Note: Spot-check the generated file before reporting numbers (YC tags can be broad), since the answers are synthetic.


## 2. Evaluation

1.Run
```bash
python main.py eval --questions data/yc/questions.jsonl \
    --models llama3.1:8b qwen2.5:14b --modes walk cypher --run-name <run>
# -> results/<run>/
# JUDGE_MODEL=gpt-4o by default (needs OPENAI_API_KEY); set JUDGE_MODEL=qwen2.5:14b for a local judge
```

The results will automatically write into `results/<run>/{raw.jsonl, per_question.csv, ragas.csv, summary.csv, system.json}`.

`<run>` is the value of `--run-name`; without it the folder is timestamped (`run-YYYYMMDD-HHMMSS`).
Name real experiments explicitly and never reuse a name: an existing folder is overwritten without warning.


All `eval` flags (a flag is an option starting with `--`; leave it out to get the default):

| Flag | Default | Effect | Trade-off |
|---|---|---|---|
| `--questions PATH` | `data/demo/questions.jsonl` | question set to evaluate | set it explicitly for the YC graph |
| `--models M [M ...]` | `llama3.1:8b qwen2.5:14b gpt-4o` | models under test | `gpt-4o` needs `OPENAI_API_KEY`; runtime grows with each model |
| `--modes X [X ...]` | `walk cypher` | retrieval: `walk`, `cypher` (Text-to-Cypher baseline), `expand` | `cypher` needs Neo4j (skipped on `memory`) |
| `--backend neo4j\|memory` | `neo4j` | `memory` runs without Docker | run `demo --backend memory` first; no `cypher`, memory holds only the synthetic demo graph |
| `--run-name NAME` | `run-YYYYMMDD-HHMMSS` | output folder `results/NAME/` | an existing folder is overwritten |
| `--limit N` | all | first N questions only (smoke test) | not a random sample; don't report these numbers |
| `--no-ragas` | off | skip the RAGAS judge: faster, no OpenAI key | RAGAS columns in `summary.csv` stay empty |
| `--no-check` | off | skip the evidence-sufficiency check (one fewer LLM call) | no follow-up retrieval when evidence is missing; recall can drop |

## 3. Project layout
```
WalkVC/
├── main.py                    # CLI: demo | collect | build | export-list | investors | questions | ask | eval  (+)
├── requirements.txt           
├── pytest.ini                 
├── .env.example               # template for .env (keys, DB passwords, model names)  
├── docker-compose.yml         # Neo4j 5.20 + MongoDB 7 for local development  
├── docker-compose.prod.yml    # production override: localhost-only ports, passwords from .env  
├── DEPLOY.md                  # deployment handoff: restore the exported data, no re-crawl  
├── tests/
│   ├── test_pipeline.py       # offline tests (no Docker / no LLM needed)
│   ├── test_anchors.py        # anchoring / pinning / intersection on a small YC-shaped graph  
│   └── test_investors.py      # funding-news harvesting and investor extraction  
├── scripts/                   
│   ├── final_rerun.py         # re-run the main configurations on the final graph (G2-final)
│   ├── answer_recall.py       # answer-level recall for every method, incl. the Cypher baseline
│   ├── explore_budget.py      # recall vs token budget + paired bootstrap + figure
│   ├── two_hop_summary.py     # pooled two-hop comparison + size-aware router simulation
│   ├── stopping_ablation.py   # info-gain stopping vs fixed / aggressive stopping
│   ├── eval_collection.py     # news-extracted investor edges vs PitchBook golden set
│   ├── collection_ceiling.py  # investor recall = source coverage x extraction recall
│   ├── check_pb_matches.py    # check YC -> PitchBook "Upload a List" matches
│   ├── backup.sh              # export Mongo + Neo4j to backups/
│   ├── restore.sh             # load a backup (overwrites local databases)
│   └── _dbutil.sh             # shared helpers for backup.sh / restore.sh
├── data/                      
│   ├── demo/                  # synthetic entities.json + questions.jsonl
│   ├── yc/                    # YC question sets: questions / questions_explore / questions_coinvest
│   └── pitchbook/             # PitchBook exports, used only by the collection-layer evaluation
├── results/                   # one folder per run; see results/README.md  
└── vcgraph/
    ├── config.py              # settings from .env
    ├── llm.py                 # unified Ollama/OpenAI client, streaming TTFT, token counting, embeddings, VRAM probe
    ├── jsonparse.py           # robust JSON parsing + parser success-rate counters (Reliability metric)
    ├── schema.py              # pydantic schemas (CompanyProfile, Decomposition, ...)
    ├── storage/
    │   ├── mongo.py           # raw/landing store
    │   └── graph.py           # Neo4jGraph + InMemoryGraph (same interface)
    ├── collect/
    │   ├── sources.py         # YC (Crawl4AI / JSON mirror), SEC EDGAR Form D, OpenCorporates
    │   ├── extract.py         # LLM page -> CompanyProfile extraction
    │   ├── investors.py       # funding news -> LLM FundingEvent -> normalized investor edges  
    │   ├── resolve.py         # entity resolution
    │   └── agent.py           # LangGraph collection pipeline
    ├── kg/
    │   ├── builder.py         # entities -> nodes/edges/embeddings (+ derived SIMILAR_TO)
    │   └── demo.py            # synthetic ecosystem + ground-truth questions
    ├── reason/
    │   ├── walk.py            # ★ guided random walk with restart + info-gain stopping
    │   ├── anchors.py         # exact-name anchoring, anchor-conditioned query, pinned neighbours 
    │   ├── decompose.py       # query decomposition / sufficiency check
    │   ├── cypher.py          # Text-to-Cypher baseline
    │   └── agent.py           # LangGraph reasoning agent
    └── eval/
        ├── make_questions.py  # real-data question sets with ground truth from YC / SEC facts 
        ├── metrics.py         # node-id precision/recall, cited ids, percentiles, GPU probe 
        └── run_eval.py        # benchmark: models x modes x questions -> RAGAS + system metrics

```
