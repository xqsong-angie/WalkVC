# Deployment handoff

This guide is for whoever deploys WalkVC. You do **not** need to re-crawl anything: the data has been exported
to two files, and `scripts/restore.sh` loads them. For how the system works, see [README.md](README.md);
this document only covers getting it running.

## 0. What you receive

| Item | Where | Notes |
|---|---|---|
| Code | GitHub `xqsong-angie/WalkVC`, tag `v1.0-report` | Every number in the report comes from this version |
| `backups/` folder | Sent separately (cloud drive or scp), **not in git** | `neo4j.dump`, `vcgraph.mongo.gz`, `counts.txt`, `SHA256SUMS` |
| Secrets | Bring your own | Your own OpenAI key (only needed for evaluation) and new database passwords |

What the two data files are for:

- **`neo4j.dump`**: the knowledge graph (nodes, edges, embeddings, vector index). Online question answering
  depends **only on this** plus Ollama.
- **`vcgraph.mongo.gz`**: the crawler's raw pages (`raw_pages`), extracted records (`records`) and the resolved
  entities (`entities`). It is not used online, but you need it to add data or rebuild the graph later, so keep it.

## 1. Server requirements

- Docker, with Docker Compose 2.24 or newer (`docker-compose.prod.yml` uses the `!override` tag)
- 16 GB RAM or more: `llama3.1:8b` takes about 9 GB, `nomic-embed-text` about 0.3 GB, and the Neo4j heap 2 GB
- A GPU is recommended. CPU-only works, but each question takes tens of seconds. To use an NVIDIA GPU,
  uncomment the `deploy` block of the `ollama` service in `docker-compose.yml`.
- Python 3.11

## 2. Restore the data

```bash
git clone https://github.com/xqsong-angie/WalkVC.git && cd WalkVC && git checkout v1.0-report
# put the backups/ folder you received in the repository root

cp .env.example .env
#   must change: NEO4J_PASSWORD, MONGO_ROOT_USER, MONGO_ROOT_PASSWORD
#   set MONGO_URI to mongodb://<user>:<pass>@localhost:27017/?authSource=admin

export COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml   # ports bound to 127.0.0.1, passwords on
scripts/restore.sh       # verifies sha256, loads Mongo and Neo4j, compares counts with the source machine
```

Success ends with `OK: counts match the source machine`. If you see `WARNING: counts differ`, stop and
contact the person who ran the experiments.

> The **first** start must use the prod config while the data volumes are still empty, because Mongo creates
> its root user only on first initialisation. If you already started the stack with the default config,
> wipe the volumes with `docker compose down -v` and then run the restore.

## 3. Models and Python environment

```bash
docker compose exec ollama ollama pull nomic-embed-text   # must be this one, see section 4
docker compose exec ollama ollama pull llama3.1:8b        # answering model
docker compose exec ollama ollama pull qwen2.5:14b        # optional: the second model in the report

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # crawl4ai-setup is not needed unless you crawl
```

## 4. Settings you must not change

| Setting | Why |
|---|---|
| `EMBED_MODEL=nomic-embed-text`, `EMBED_DIM=768` | Every node vector in the graph was computed with this model, and questions are encoded with it at query time. With a different model, vector search returns meaningless results. |
| `WALK_*` parameters | The point and explore experiments in the report used `WALK_TOP_NODES=12` with the other values at their defaults (see `.env.example`). Changing them makes results differ from the report. |
| Only the `neo4j` database is dumped and loaded | Users and passwords live in the `system` database, so the server's own `NEO4J_PASSWORD` applies, not anything in the dump. |

What you may change: `DEFAULT_MODEL`, the LLM used for answering. A larger local model or an API model both
work; the answers will just differ from the report.

## 5. Acceptance check

```bash
EMBED_BACKEND=hash pytest -q                         # offline unit tests, no database needed
python main.py ask "<any question from data/yc/questions.jsonl>" -v
```

`-v` prints the evidence subgraph. Compare 2–3 questions with the output on the experiment machine: the seed
nodes and the entities in the evidence should largely match. The wording of the LLM's answer will vary between
runs; that is expected.

## 6. Serving it

The project currently has **a CLI only, no HTTP API**. To build a service, the entry point is `ask()` in
`vcgraph/reason/agent.py`:

```python
from vcgraph.reason.agent import ask
from vcgraph.storage.graph import Neo4jGraph

g = Neo4jGraph()                      # create one per process and reuse it
r = ask("Who has co-invested alongside ...?", g, mode="walk", on_token=None)
r["answer"], r["contexts"], r["metrics"]["e2e_latency_s"]
```

- `on_token` is a callback invoked for each generated token; wire it to SSE or a WebSocket for streaming.
- `mode` is `walk` (this project's method), `cypher` (Text-to-Cypher baseline) or `expand` (k-hop baseline).
  Use `walk` in production.
- Neo4j, Mongo and Ollama listen on 127.0.0.1 only; do not open them to the internet. Expose only your own API.

## 7. Adding data or rebuilding the graph later

- **Rebuild Neo4j from Mongo** (no crawling, but Ollama recomputes the embeddings): `python main.py build`
- **Add investor data**: `python main.py investors ...`
- **Crawl new companies**: `python main.py collect ...`. Run `crawl4ai-setup` first and set `SEC_USER_AGENT`
  (SEC requires a real contact email) and `OPENCORPORATES_API_TOKEN` in `.env`.

After any data change, run `scripts/backup.sh` to make a fresh backup.

**Open design decision:** the collection agent and the reasoning agent are not connected, so a question about a
company that is not in the graph cannot trigger collection. Whether to close that loop is up to you; see
[README § 6 TODO](README.md#6-todo-for-deployment) for what it would take.

## 8. Compliance

The data in `data/pitchbook/` comes from the university's PitchBook licence. Before showing results based on it
publicly, confirm that the licence terms allow it.
