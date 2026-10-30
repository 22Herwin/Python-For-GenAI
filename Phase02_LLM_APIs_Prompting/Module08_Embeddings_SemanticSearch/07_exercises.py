# Module 08 - Embeddings & Semantic Search
# 8.8 Module 08 Exercises
#
# Every exercise here accepts an injectable `embed_fn` so the core logic
# (duplicate detection, hybrid scoring, index consistency, caching) can be
# fully tested offline with a deterministic mock embedder - the same pattern
# used for the OpenAI-backed pipelines elsewhere in this module. Swap in the
# real embed_fn (OpenAI's text-embedding-3-small) when you have an API key.

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv

from _openrouter_client import get_openai_client, EMBEDDING_MODEL

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

EmbedFn = Callable[[list[str]], np.ndarray]


def mock_embed(texts: list[str], dim: int = 16) -> np.ndarray:
    """Deterministic hash-based mock embedding - same text always maps to the
    same vector, with no network call. Good enough to test logic offline;
    not semantically meaningful like a real embedding model."""
    vectors = [
        np.random.default_rng(abs(hash(text)) % 2**31).standard_normal(dim)
        for text in texts
    ]
    return np.array(vectors, dtype=np.float32)


def real_openai_embed(texts: list[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    """The real embedder from section 8.2 - needs OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY."""
    client = get_openai_client()
    resp = client.embeddings.create(input=texts, model=model)
    vectors = sorted(resp.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Implement a DuplicateDetector class that takes a list of documents, embeds
# them, and returns pairs with cosine similarity above a configurable
# threshold. Use it to find near-duplicate entries in a corpus.

class DuplicateDetector:
    def __init__(self, embed_fn: EmbedFn = mock_embed, threshold: float = 0.95):
        self.embed_fn = embed_fn
        self.threshold = threshold

    def find_duplicates(self, texts: list[str]) -> list[tuple[int, int, float]]:
        """Return (index_a, index_b, score) for every pair whose cosine
        similarity is >= self.threshold."""
        embeddings = _normalise(self.embed_fn(texts))
        sim_matrix = embeddings @ embeddings.T

        pairs = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= self.threshold:
                    pairs.append((i, j, score))
        return sorted(pairs, key=lambda p: p[2], reverse=True)


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Build a HybridSearch class that combines semantic similarity (embedding
# cosine score) with a simple keyword score. alpha blends the two:
# final_score = alpha * semantic + (1 - alpha) * keyword

def _keyword_score(query: str, text: str) -> float:
    """Simple keyword overlap score in [0, 1]: fraction of query words that
    appear in the text. A lightweight stand-in for a real BM25 scorer."""
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


class HybridSearch:
    def __init__(self, embed_fn: EmbedFn = mock_embed, alpha: float = 0.5):
        self.embed_fn = embed_fn
        self.alpha = alpha
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None

    def index(self, texts: list[str]) -> None:
        self._texts = texts
        self._embeddings = _normalise(self.embed_fn(texts))

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        q_vec = _normalise(self.embed_fn([query]))[0]
        semantic_scores = self._embeddings @ q_vec
        keyword_scores = np.array([_keyword_score(query, t) for t in self._texts])

        final_scores = self.alpha * semantic_scores + (1 - self.alpha) * keyword_scores
        top_idx = np.argsort(final_scores)[::-1][:k]
        return [(int(i), float(final_scores[i])) for i in top_idx]


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Extend VectorStore to support delete(doc_id) and update(doc_id, new_text).
# Ensure the internal matrix stays consistent after each operation.

class MutableVectorStore:
    """Same idea as the VectorStore in 03_vector_store_semantic_search.py,
    extended with delete() and update() that keep the internal matrix and
    document list in sync."""

    def __init__(self, embed_fn: EmbedFn = mock_embed):
        self.embed_fn = embed_fn
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None

    def add(self, doc_id: str, text: str) -> None:
        vec = _normalise(self.embed_fn([text]))
        self._ids.append(doc_id)
        self._texts.append(text)
        self._matrix = vec if self._matrix is None else np.vstack([self._matrix, vec])

    def delete(self, doc_id: str) -> bool:
        """Remove a document by id. Returns True if it was found and removed."""
        if doc_id not in self._ids:
            return False
        idx = self._ids.index(doc_id)
        del self._ids[idx]
        del self._texts[idx]
        self._matrix = np.delete(self._matrix, idx, axis=0) if self._matrix is not None else None
        return True

    def update(self, doc_id: str, new_text: str) -> bool:
        """Re-embed a document in place. Returns True if it was found."""
        if doc_id not in self._ids:
            return False
        idx = self._ids.index(doc_id)
        self._texts[idx] = new_text
        new_vec = _normalise(self.embed_fn([new_text]))[0]
        self._matrix[idx] = new_vec
        return True

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        q_vec = _normalise(self.embed_fn([query]))[0]
        scores = self._matrix @ q_vec
        k = min(k, len(self._ids))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    @property
    def size(self) -> int:
        return len(self._ids)


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Implement embed_with_cache: a wrapper around the embeddings API that stores
# results in a local SQLite database (keyed by SHA-256 of text + model name)
# so repeated calls for the same text never hit the API twice.

class EmbeddingCache:
    def __init__(self, db_path: str, embed_fn: EmbedFn = mock_embed, model_name: str = "mock"):
        self.embed_fn = embed_fn
        self.model_name = model_name
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "hash TEXT PRIMARY KEY, text TEXT, model TEXT, vector BLOB, dim INTEGER)"
        )
        self._conn.commit()
        self.api_calls_made = 0   # for demonstrating cache hits vs misses

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}:{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        results: dict[int, np.ndarray] = {}
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            row = self._conn.execute(
                "SELECT vector, dim FROM embeddings WHERE hash = ?", (self._key(text),)
            ).fetchone()
            if row is not None:
                vector_blob, dim = row
                results[i] = np.frombuffer(vector_blob, dtype=np.float32).reshape(dim)
            else:
                to_fetch.append((i, text))

        if to_fetch:
            self.api_calls_made += 1   # one batched call for all cache misses
            fetched = self.embed_fn([t for _, t in to_fetch])
            for (i, text), vector in zip(to_fetch, fetched):
                vector = vector.astype(np.float32)
                results[i] = vector
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings (hash, text, model, vector, dim) VALUES (?,?,?,?,?)",
                    (self._key(text), text, self.model_name, vector.tobytes(), vector.shape[0]),
                )
            self._conn.commit()

        return np.array([results[i] for i in range(len(texts))], dtype=np.float32)

    def close(self) -> None:
        self._conn.close()


if __name__ == "__main__":
    print("=== Exercise 1: DuplicateDetector (offline, mock embeddings) ===")
    detector = DuplicateDetector(threshold=0.0)   # threshold=0 just to show pair scores
    corpus = [
        "What is RAG?",
        "What is RAG?",              # exact duplicate of the first
        "Explain retrieval augmented generation.",
        "What is the capital of France?",
    ]
    for i, j, score in detector.find_duplicates(corpus)[:3]:
        print(f"  [{i}] vs [{j}]: {score:.4f}  ->  {corpus[i]!r} | {corpus[j]!r}")

    print("\n=== Exercise 2: HybridSearch (offline, mock embeddings) ===")
    hybrid = HybridSearch(alpha=0.5)
    hybrid.index([
        "RAG combines retrieval and generation",
        "Vector databases store embeddings",
        "Python is used for machine learning",
    ])
    for idx, score in hybrid.search("retrieval and generation", k=2):
        print(f"  [{idx}] score={score:.4f}: {hybrid._texts[idx]}")

    print("\n=== Exercise 3: MutableVectorStore delete/update (offline) ===")
    mstore = MutableVectorStore()
    mstore.add("d1", "RAG connects LLMs to external knowledge.")
    mstore.add("d2", "Vector databases enable fast similarity search.")
    mstore.add("d3", "Fine-tuning updates model weights.")
    print("Size after adds:", mstore.size)
    mstore.delete("d2")
    print("Size after delete:", mstore.size, "| ids:", mstore._ids)
    mstore.update("d1", "RAG retrieves documents and injects them into context.")
    print("Updated text for d1:", mstore._texts[mstore._ids.index("d1")])

    print("\n=== Exercise 4: EmbeddingCache (offline, mock embeddings) ===")
    cache_path = DATA_DIR / "embedding_cache.db"
    if cache_path.exists():
        cache_path.unlink()   # fresh demo each run
    cache = EmbeddingCache(str(cache_path))
    texts_to_embed = ["hello world", "hello world", "a different sentence"]
    vectors = cache.embed(texts_to_embed)
    print(f"Embedded {len(texts_to_embed)} texts, shape={vectors.shape}, "
          f"api_calls_made={cache.api_calls_made} (only 1 call for 2 unique texts)")
    # Call again with the same texts - should hit the cache, zero new API calls
    cache.embed(texts_to_embed)
    print(f"After re-embedding the same texts: api_calls_made={cache.api_calls_made} (unchanged)")
    cache.close()
