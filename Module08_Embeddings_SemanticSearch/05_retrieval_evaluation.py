# Module 08 - Embeddings & Semantic Search
# 8.6 Evaluating Retrieval Quality
# Before connecting retrieval to a generator, verify that the right documents
# are actually being retrieved.
#
# precision_at_k, recall_at_k, and mean_reciprocal_rank are pure Python - no
# API key needed. evaluate_retrieval() needs a real VectorStore (built with
# OpenAI embeddings in 03_vector_store_semantic_search.py), so its demo is
# gated behind an OPENAI_API_KEY check.

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from _openrouter_client import has_openai_access

load_dotenv()


@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]   # ground-truth relevant documents


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(store, eval_cases: list[RetrievalEvalCase], k: int = 5) -> dict:
    """Run all eval cases and return aggregate metrics."""
    p_scores, r_scores, mrr_scores = [], [], []

    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [r.document.id for r in results]

        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))

    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 4),
        f"recall@{k}":    round(float(np.mean(r_scores)), 4),
        "MRR":            round(float(np.mean(mrr_scores)), 4),
    }


def _load_vector_store_module():
    """Dynamically import 03_vector_store_semantic_search.py (its filename
    starts with a digit, so a normal `import` statement can't reach it)."""
    path = Path(__file__).parent / "03_vector_store_semantic_search.py"
    spec = importlib.util.spec_from_file_location("vector_store_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    print("=== Offline: precision_at_k / recall_at_k / MRR on mock retrieval results ===")
    mock_retrieved = ["d08", "d03", "d01", "d05"]
    mock_relevant = ["d01", "d08"]
    print("precision@3:", precision_at_k(mock_retrieved, mock_relevant, k=3))
    print("recall@3:   ", recall_at_k(mock_retrieved, mock_relevant, k=3))
    print("MRR:        ", mean_reciprocal_rank(mock_retrieved, mock_relevant))

    print("\n=== evaluate_retrieval() against a real VectorStore (needs OPENAI_API_KEY) ===")
    if not has_openai_access():
        print("Skipped - no OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY set.")
    else:
        vs_module = _load_vector_store_module()
        store = vs_module.VectorStore()
        store.add_documents(vs_module.CORPUS)

        eval_cases = [
            RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
            RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
            RetrievalEvalCase("How do agents use language models?", ["d10"]),
            RetrievalEvalCase("What is fine-tuning?", ["d03"]),
            RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
        ]
        metrics = evaluate_retrieval(store, eval_cases, k=3)
        print(metrics)
