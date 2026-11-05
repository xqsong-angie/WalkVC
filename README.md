# Agentic KG Exploration via Random Walks for VC Market Intelligence

```
                 ┌───────────────────── Data Collection Agent (LangGraph) ─────────────────────┐
 YC directory ──▶│ Crawl4AI render ─▶ LLM extraction ─┐                                        │
 SEC EDGAR    ──▶│ Form D search + XML parse ─────────┼─▶ MongoDB (raw_pages, records)          │
 OpenCorporates─▶│ legal-entity search ───────────────┘        │                               │
                 │                              entity resolution (fuzzy + domain + union-find)│
                 └──────────────────────────────────────────────┼──────────────────────────────┘
                                                                ▼
                                   Neo4j 5 (KG structure + vector index on :Entity.embedding)
                                                                ▲
                 ┌───────────────────── Graph Reasoning Agent (LangGraph) ─────────────────────┐
  NL question ──▶│ decompose ─▶ seed (vector + name) ─▶ guided random walks ─▶ subgraph context  │
                 │      ▲                                   (stop on marginal info gain)  │      │
                 │      └──────── follow-up sub-query ◀── sufficiency check ◀─────────────┘      │
                 │                                                     ─▶ streamed answer + [citations]
                 └──────────────────────────────────────────────────────────────────────────────┘
  Baseline: mode=cypher  (Text-to-Cypher deterministic retrieval)
```

## 1. Project layout

```
main.py                    CLI: demo | collect | build | ask | eval
vcgraph/config.py          settings from .env
vcgraph/llm.py             unified Ollama/OpenAI client, streaming TTFT, token counting, embeddings, VRAM probe
vcgraph/jsonparse.py       robust JSON parsing + parser success-rate counters (Reliability metric)
vcgraph/schema.py          pydantic schemas (CompanyProfile, Decomposition, ...)
vcgraph/storage/mongo.py   raw/landing store
vcgraph/storage/graph.py   Neo4jGraph + InMemoryGraph (same interface)
vcgraph/collect/sources.py YC (Crawl4AI / JSON mirror), SEC EDGAR Form D, OpenCorporates
vcgraph/collect/extract.py LLM page -> CompanyProfile extraction
vcgraph/collect/resolve.py entity resolution
vcgraph/collect/agent.py   LangGraph collection pipeline
vcgraph/kg/builder.py      entities -> nodes/edges/embeddings (+ derived SIMILAR_TO)
vcgraph/kg/demo.py         synthetic ecosystem + ground-truth questions
vcgraph/reason/walk.py     ★ guided random walk with restart + info-gain stopping
vcgraph/reason/decompose.py query decomposition / sufficiency check
vcgraph/reason/cypher.py   Text-to-Cypher baseline
vcgraph/reason/agent.py    LangGraph reasoning agent
vcgraph/eval/run_eval.py   benchmark: models x modes x questions -> RAGAS + system metrics
tests/test_pipeline.py     offline tests (no Docker / no LLM needed)
```

## 2. Quick start

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

### 2.1 Option A — synthetic demo graph (fastest way to get end-to-end numbers)

```bash
python main.py demo                         # 72 fictional companies -> Neo4j, writes data/demo/questions.jsonl
python main.py ask "Which startups compete with Junostack?" -v
python main.py ask "Who has co-invested alongside Northpeak Ventures?" --model qwen2.5:14b
python main.py ask "Which investors are most active in Fintech startups?" --mode cypher   # baseline
```

Company names come from `data/demo/questions.jsonl` (they are generated deterministically).
No Docker? `python main.py demo --backend memory` and add `--backend memory` to `ask` / `eval`
(Cypher mode needs Neo4j).

### 2.2 Option B — real Y Combinator data (replaces the synthetic demo)

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

# 3) ask / evaluate on the real graph
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
| network | Which companies is P connected to? | person appears at ≥2 companies (founder / Form D officer) |
| investors | Which investors backed X? | only if investors were extracted from pages |

Spot-check the generated file before reporting numbers (YC tags can be broad).
Run `python main.py demo` only if you still want the synthetic graph; it writes to the same Neo4j,
so clear it first: `MATCH (n) DETACH DELETE n` in the Neo4j browser (http://localhost:7474).

### 3. Evaluation

```bash
python main.py eval --models llama3.1:8b qwen2.5:14b gpt-4o --modes walk cypher
# JUDGE_MODEL=gpt-4o by default (needs OPENAI_API_KEY); set JUDGE_MODEL=qwen2.5:14b for a local judge
```

Writes `results/<run>/{raw.jsonl, per_question.csv, ragas.csv, summary.csv, system.json}`.

| Dimension | Columns in `summary.csv` |
|---|---|
| Task quality | `faithfulness`, `hallucination_rate` (=1−faithfulness), `answer_relevancy`, `context_precision`, `factual_correctness`, `retrieval_recall` / `citation_precision` (against reference node ids) |
| Latency | `ttft_p50_s`, `e2e_p50_s`, `e2e_p95_s`, `walk_iter_p95_s`, `walk_iter_under_3s` |
| Resource use | `tokens_mean`, `context_tokens_mean` vs `full_context_tokens_mean` (2-hop dump), `token_savings_vs_full`, `vram_mb` (Ollama `/api/ps`), `gpu_used_mb` |
| Reliability | `json_parse_rate` (all structured LLM outputs), `cypher_exec_rate`, `cypher_first_try_rate` |

## 4. The random walk (vcgraph/reason/walk.py)

Transition from `u` to neighbour `v` for sub-query `q`:

```
P(v|u,q) ∝ exp( [ α·cos(q, e_v) + β·log w_rel(type) + γ·log w_label(label(v))
                  − λ·log(1+deg v) − μ·1[v = previous] ] / T ),   restart to seeds with p = 0.15
```

* `w_rel`, `w_label` come from the intent and the `relation_hints` / `target_types` of the LLM decomposition
  (e.g. *competitors* favours `SIMILAR_TO`, `IN_SECTOR`; *founders* favours `FOUNDED`, `WORKED_AT`).
* Walkers move in lockstep: 2 batched DB round-trips per iteration whatever the number of walkers;
  only the touched neighbourhood is cached (memory-efficient).
* **Stopping on marginal information gain**: `IG_t = rel-mass of nodes newly entering the top-K evidence set / rel-mass of the set`.
  Stop when `IG_t < ε` for `patience` iterations (defaults `ε=0.05`, `patience=2`, `max_iters=12`).
* All knobs are configurable through `.env` (`WALK_*`) or `WalkParams`.

On the synthetic graph (hash embeddings, no LLM), the walk stops after 4–8 iterations on most questions,
recalls the reference nodes, and uses ~80% fewer context tokens than a 2-hop neighbourhood dump.

## 5. Limitationstes

* The YC directory is JS-rendered. `hybrid` / `crawl` use Crawl4AI (3 concurrent pages). Keep crawls
  small and for research use; check YC's terms. If YC blocks headless browsers, use `--yc-mode json`.
* `yc-oss.github.io` is a community mirror updated periodically, so the newest batch may lag.
* SEC: respect fair access (≤10 req/s); a descriptive `SEC_USER_AGENT` with an email is required.
  Form D full-text matches can be noisy: entity resolution keeps only clusters anchored to a YC company.
* OpenCorporates needs `OPENCORPORATES_API_TOKEN`; otherwise that step is skipped.
* Form D does not list investors, so `INVESTED_IN` edges in real data only come from investors the LLM
  extracts from crawled pages; the demo graph has synthetic investors.
* Measure TTFT after the warm-up call (`run_eval` does this) so model load time is excluded.
