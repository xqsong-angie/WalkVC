"""Benchmark runner: models x retrieval modes x questions -> RAGAS + system metrics.

Outputs (under results/<run_name>/):
  raw.jsonl     one line per (model, mode, question) with answer, contexts, telemetry
  ragas.csv     per-sample RAGAS scores
  summary.csv   one row per (model, mode) across all four evaluation dimensions
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..config import settings
from ..jsonparse import parser_stats
from ..llm import count_tokens, is_openai_model, ollama_vram
from ..reason.agent import ask
from ..reason.walk import full_context_baseline
from ..storage.graph import GraphBackend
from .metrics import cited_ids, id_precision_recall, mean, nvidia_smi_used_mb, pct


def load_questions(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _judge():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    jm = settings.judge_model
    if is_openai_model(jm):
        llm = ChatOpenAI(model=jm, api_key=settings.openai_api_key, temperature=0)
    else:
        llm = ChatOpenAI(model=jm, base_url=f"{settings.ollama_host}/v1", api_key="ollama", temperature=0)
    if settings.openai_api_key:
        emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    else:
        emb = OpenAIEmbeddings(model=settings.embed_model, base_url=f"{settings.ollama_host}/v1",
                               api_key="ollama", check_embedding_ctx_length=False)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def run_ragas(rows: list[dict]) -> pd.DataFrame:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference,
                               ResponseRelevancy)

    llm, emb = _judge()
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "retrieved_contexts": r["contexts"] or ["(no context retrieved)"],
         "response": r["answer"] or "(no answer)", "reference": r["reference"]}
        for r in rows
    ])
    res = evaluate(dataset=ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                                        FactualCorrectness()],
                   llm=llm, embeddings=emb, raise_exceptions=False, show_progress=True)
    df = res.to_pandas()
    for k in ("model", "mode", "qid"):
        df[k] = [r[k] for r in rows]
    return df


def run_benchmark(g: GraphBackend, questions_path: str, models: list[str], modes: list[str],
                  out_dir: str = "results", run_name: str | None = None, ragas: bool = True,
                  check: bool = True, limit: int | None = None) -> pd.DataFrame:
    qs = load_questions(questions_path)[:limit]
    run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
    out = Path(out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    rows, sys_rows = [], []

    for model in models:
        # warm-up so model load time does not pollute TTFT of the first question
        try:
            ask("warm-up: list one company", g, model=model, mode="walk", check=False)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] warm-up failed for {model}: {e}")
        parser_stats(reset=True)
        vram = ollama_vram() if not is_openai_model(model) else {}
        gpu_mb = nvidia_smi_used_mb()

        for mode in modes:
            if mode == "cypher" and not hasattr(g, "driver"):
                print("[eval] skipping cypher mode (needs Neo4j backend)")
                continue
            for q in qs:
                t0 = time.perf_counter()
                try:
                    r = ask(q["question"], g, model=model, mode=mode, check=check)
                except Exception as e:  # noqa: BLE001
                    print(f"[eval] {model}/{mode}/{q['id']} failed: {e}")
                    r = {"answer": "", "contexts": [], "retrieved_ids": [], "seed_ids": [],
                         "metrics": {"e2e_latency_s": time.perf_counter() - t0, "error": str(e)}}
                m = r["metrics"]
                ids = r["retrieved_ids"] if mode in ("walk", "expand") else []
                cited = cited_ids(r["answer"])
                pr_ret = id_precision_recall(ids, q.get("reference_ids", []))
                pr_cit = id_precision_recall(cited, q.get("reference_ids", []))
                full_tok = None
                if mode == "walk" and r.get("seed_ids"):
                    full_tok = count_tokens(full_context_baseline(g, r["seed_ids"], hops=2))
                walks = m.get("walks", []) or []
                iter_lat = [x for w in walks for x in w.get("iter_latency_s", [])]
                row = {
                    "model": model, "mode": mode, "qid": q["id"], "qtype": q.get("type"),
                    "question": q["question"], "reference": q["reference"],
                    "answer": r["answer"], "contexts": r["contexts"],
                    "retrieval_precision": pr_ret["precision"], "retrieval_recall": pr_ret["recall"],
                    "citation_precision": pr_cit["precision"], "citation_recall": pr_cit["recall"],
                    "ttft_s": m.get("ttft_s"), "e2e_latency_s": m.get("e2e_latency_s"),
                    "walk_time_s": (m.get("timings") or {}).get("walk"),
                    "walk_iters": sum(w.get("iterations", 0) for w in walks),
                    "walk_iter_p95_s": pct(iter_lat, 95), "walk_iter_max_s": max(iter_lat) if iter_lat else None,
                    "total_tokens": (m.get("llm") or {}).get("total_tokens"),
                    "context_tokens": m.get("context_tokens"), "full_context_tokens": full_tok,
                    "cypher_valid": (m.get("cypher") or {}).get("error") is None if mode == "cypher" else None,
                    "cypher_first_try": (m.get("cypher") or {}).get("first_try_valid") if mode == "cypher" else None,
                    "cypher_non_empty": (m.get("cypher") or {}).get("non_empty") if mode == "cypher" else None,
                                        "expand_truncated": (m.get("expand") or {}).get("truncated") if mode == "expand" else None,
                    "expand_full_tokens_est": (m.get("expand") or {}).get("est_full_tokens") if mode == "expand" else None,
                    "metrics": m,
                }
                rows.append(row)
                print(f"[eval] {model:14s} {mode:6s} {q['id']:28s} e2e={row['e2e_latency_s']:.2f}s "
                      f"ttft={row['ttft_s']} R={row['retrieval_recall']}")
        ps = parser_stats()
        json_att = sum(v["attempts"] for k, v in ps.items() if not k.startswith("cypher"))
        json_ok = sum(v["success"] for k, v in ps.items() if not k.startswith("cypher"))
        sys_rows.append({"model": model, "vram": vram, "gpu_used_mb": gpu_mb, "parser": ps,
                         "json_parse_rate": round(json_ok / json_att, 4) if json_att else None})

    with open(out / "raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (out / "system.json").write_text(json.dumps(sys_rows, indent=2, default=str))

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("contexts", "metrics")} for r in rows])
    if ragas and rows:
        try:
            rg = run_ragas(rows)
            rg.to_csv(out / "ragas.csv", index=False)
            score_cols = [c for c in rg.columns if c not in ("user_input", "retrieved_contexts", "response",
                                                              "reference", "model", "mode", "qid")]
            df = df.merge(rg[["model", "mode", "qid", *score_cols]], on=["model", "mode", "qid"], how="left")
        except Exception as e:  # noqa: BLE001
            print(f"[eval] RAGAS failed: {e}")

    summary = summarize(df, sys_rows)
    df.to_csv(out / "per_question.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"\n[eval] wrote {out}/\n")
    print(summary.to_string(index=False))
    return summary


def summarize(df: pd.DataFrame, sys_rows: list[dict]) -> pd.DataFrame:
    sysmap = {s["model"]: s for s in sys_rows}
    out = []
    def col(d, prefix):  # RAGAS sometimes suffixes names, e.g. factual_correctness(mode=f1)
        c = next((c for c in d.columns if c.startswith(prefix)), None)
        return list(d[c]) if c else []

    for (model, mode), d in df.groupby(["model", "mode"]):
        s = sysmap.get(model, {})
        vram = s.get("vram") or {}
        row = {
            "model": model, "mode": mode, "n": len(d),
            # --- task quality
            "faithfulness": mean(col(d, "faithfulness")),
            "hallucination_rate": (round(1 - mean(col(d, "faithfulness")), 4)
                                   if mean(col(d, "faithfulness")) is not None else None),
            "answer_relevancy": mean(col(d, "answer_relevancy")),
            "context_precision": mean(col(d, "llm_context_precision")),
            "factual_correctness": mean(col(d, "factual_correctness")),
                        "retrieval_recall": mean(d["retrieval_recall"]) if mode in ("walk", "expand") else None,
            # --- latency
            "ttft_p50_s": pct(d["ttft_s"], 50), "e2e_p50_s": pct(d["e2e_latency_s"], 50),
            "e2e_p95_s": pct(d["e2e_latency_s"], 95), "walk_iter_p95_s": pct(d["walk_iter_p95_s"], 95),
            "walk_iter_under_3s": mean([(x is not None and x < 3.0) for x in d["walk_iter_max_s"]]) if mode == "walk" else None,
            # --- resource use
            "tokens_mean": mean(d["total_tokens"]), "context_tokens_mean": mean(d["context_tokens"]),
            "full_context_tokens_mean": mean(d["full_context_tokens"]),
            "vram_mb": next((v.get("size_vram_mb") for k, v in vram.items() if isinstance(v, dict) and k.startswith(model.split(":")[0])), None),
            "gpu_used_mb": s.get("gpu_used_mb"),
            # --- reliability
            "json_parse_rate": s.get("json_parse_rate"),
            "cypher_exec_rate": mean([float(x) for x in d["cypher_valid"] if x is not None]) if mode == "cypher" else None,
            "cypher_first_try_rate": mean([float(x) for x in d["cypher_first_try"] if x is not None]) if mode == "cypher" else None,
            "cypher_non_empty_rate": mean([float(x) for x in d["cypher_non_empty"] if x is not None]) if mode == "cypher" else None,
                        "expand_truncated_rate": mean([float(x) for x in d["expand_truncated"] if x is not None]) if mode == "expand" else None,
            "expand_full_tokens_est": mean(d["expand_full_tokens_est"]) if mode == "expand" else None,
        }
        if row["full_context_tokens_mean"] and row["context_tokens_mean"]:
            row["token_savings_vs_full"] = round(1 - row["context_tokens_mean"] / row["full_context_tokens_mean"], 4)
        out.append(row)
    return pd.DataFrame(out)
