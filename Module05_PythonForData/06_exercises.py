# Module 05 - Python for Data
# 5.4 Module 05 Exercises
#
# Fill in each function below. Run this file directly to test your solutions
# against the sample data provided.

import numpy as np
import pandas as pd
from pathlib import Path


def benchmark_summary(csv_path: str) -> dict:
    """
    Exercise 1:
    Load a CSV of LLM benchmark scores (create one with at least 20 rows and
    4 models, columns like: model, task, score, latency_ms). Compute:
      - mean score per model
      - best-performing task per model
      - correlation between score and latency columns
    Return a dict with keys "mean_score_per_model", "best_task_per_model",
    and "score_latency_corr".
    """
    df = pd.read_csv(csv_path)

    mean_score_per_model = df.groupby("model")["score"].mean().round(4).to_dict()

    # Best task per model = task with the highest average score, per model
    task_avg = df.groupby(["model", "task"])["score"].mean()
    best_task_per_model = task_avg.groupby("model").idxmax().apply(lambda t: t[1]).to_dict()

    score_latency_corr = round(float(df["score"].corr(df["latency_ms"])), 4)

    return {
        "mean_score_per_model": mean_score_per_model,
        "best_task_per_model": best_task_per_model,
        "score_latency_corr": score_latency_corr,
    }


def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    """
    Exercise 2:
    Write a function normalise_embeddings(matrix) -> np.ndarray that
    L2-normalises each row. Verify that all row norms equal 1.0 after
    normalisation.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return matrix / norms


def text_folder_stats(folder_path: str) -> pd.DataFrame:
    """
    Exercise 3:
    Implement a function that reads a folder of .txt files and returns a
    Pandas DataFrame with columns: filename, char_count, word_count,
    sentence_count. Sort by word_count descending.
    """
    import re

    rows = []
    for path in Path(folder_path).glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
        rows.append({
            "filename": path.name,
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("word_count", ascending=False).reset_index(drop=True)


def most_similar_pair(texts: list[str], dim: int = 16) -> tuple[int, int, float]:
    """
    Exercise 4:
    Build a pairwise cosine similarity matrix for a small corpus of 5 strings
    using hash-based mock embeddings. Find and return the pair with the
    highest similarity as (index_a, index_b, score).
    """
    # Deterministic hash-based mock embeddings (same text -> same vector)
    embeddings = np.array([
        np.random.default_rng(abs(hash(text)) % 2**31).standard_normal(dim)
        for text in texts
    ])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1, norms)

    sim_matrix = normed @ normed.T  # (n, n) pairwise cosine similarities

    n = len(texts)
    best = (-1, -1, -1.0)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score > best[2]:
                best = (i, j, score)

    return best


if __name__ == "__main__":
    # ── Sample data setup for Exercise 1 ─────────────────────────────────────
    rng = np.random.default_rng(7)
    models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro", "llama-3.1-70b"]
    tasks = ["qa", "summarise", "code", "reasoning"]

    rows = []
    for i in range(24):
        model = models[i % len(models)]
        task = tasks[i % len(tasks)]
        score = round(float(rng.uniform(0.6, 0.98)), 3)
        latency = round(float(rng.uniform(200, 800)), 1)
        rows.append({"model": model, "task": task, "score": score, "latency_ms": latency})

    sample_csv = Path(__file__).parent / "benchmark_scores.csv"
    pd.DataFrame(rows).to_csv(sample_csv, index=False)

    # ── Sample data for Exercise 2 ────────────────────────────────────────────
    sample_matrix = rng.standard_normal((5, 8))

    # ── Sample data for Exercise 3 ────────────────────────────────────────────
    sample_text_dir = Path(__file__).parent / "sample_texts"
    sample_text_dir.mkdir(exist_ok=True)
    (sample_text_dir / "doc1.txt").write_text(
        "RAG combines retrieval and generation. It is powerful. It reduces hallucination.",
        encoding="utf-8",
    )
    (sample_text_dir / "doc2.txt").write_text(
        "Python is a great language for AI. It has NumPy and Pandas.",
        encoding="utf-8",
    )

    # ── Sample data for Exercise 4 ────────────────────────────────────────────
    sample_texts = [
        "What is RAG?",
        "Explain retrieval augmented generation.",
        "What is the capital of France?",
        "Tell me about Paris, France.",
        "How do vector databases work?",
    ]

    print(benchmark_summary(str(sample_csv)))

    normed = normalise_embeddings(sample_matrix)
    print("Row norms after normalisation:", np.linalg.norm(normed, axis=1))  # should all be ~1.0

    print(text_folder_stats(str(sample_text_dir)))

    idx_a, idx_b, score = most_similar_pair(sample_texts)
    print(f"Most similar pair: [{idx_a}] {sample_texts[idx_a]!r} <-> "
          f"[{idx_b}] {sample_texts[idx_b]!r} (score={score:.4f})")
