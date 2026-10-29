# Module 11 - Vector Databases
# 11.6 Module 11 Exercises
#
# Every exercise here uses ChromaDB with an in-memory client (chromadb.EphemeralClient)
# and an injectable `embed_fn`, so the whole file runs offline with the
# deterministic mock embedder from _openrouter_client.py. Swap in openai_embed
# (text-embedding-3-small) and/or a PersistentClient when you want the real thing.
#
# NOTE: Needs the `chromadb` package (pip install chromadb). Embeddings are
# offline by default (no API key needed). The JSON export from Exercise 4 is
# written to ./data/chroma_export.json in this folder.

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from _openrouter_client import DATA_DIR, EmbedFn, mock_embed

try:
    import chromadb
except ImportError:   # handled gracefully in __main__
    chromadb = None

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection


# ── Shared helpers ─────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    """One ranked result. `score` is higher-is-better (see _distance_to_score)."""
    collection: str
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_chromadb() -> None:
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Run: pip install chromadb")


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_cosine_collection(client: ClientAPI, name: str) -> Collection:
    """Get or create a collection that ranks by cosine distance (same setting as 11.2)."""
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _space_of(collection: Collection) -> str:
    """Distance space of a collection; ChromaDB's default is l2 when none was set."""
    return (collection.metadata or {}).get("hnsw:space", "l2")


def _distance_to_score(distance: float, space: str) -> float:
    """Turn a ChromaDB distance (lower = closer) into a higher-is-better score so
    results from different collections can be merged on one scale."""
    if space in ("cosine", "ip"):
        return 1.0 - distance       # cosine / inner-product distance = 1 - similarity
    return 1.0 / (1.0 + distance)   # l2 distance -> (0, 1]


def _add_records(
    collection: Collection,
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any] | None] | None = None,
    upsert: bool = False,
) -> None:
    """add()/upsert() a batch. ChromaDB rejects empty metadata dicts, so records
    without metadata are written in a separate call with no `metadatas`."""
    write = collection.upsert if upsert else collection.add
    metadatas = metadatas or [None] * len(ids)

    with_meta = [i for i, m in enumerate(metadatas) if m]
    without_meta = [i for i, m in enumerate(metadatas) if not m]

    for group, has_meta in ((with_meta, True), (without_meta, False)):
        if not group:
            continue
        kwargs: dict[str, Any] = dict(
            ids=[ids[i] for i in group],
            documents=[texts[i] for i in group],
            embeddings=[embeddings[i] for i in group],
        )
        if has_meta:
            kwargs["metadatas"] = [metadatas[i] for i in group]
        write(**kwargs)


def _query_hits(
    collection: Collection,
    q_vec: list[float],
    k: int,
    where: dict[str, Any] | None = None,
) -> list[SearchHit]:
    """Query one collection and return scored hits (empty collection -> no hits)."""
    n = collection.count()
    if n == 0:
        return []
    space = _space_of(collection)
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=min(k, n),
        where=where,
        include=["documents", "distances", "metadatas"],
    )
    return [
        SearchHit(
            collection=collection.name,
            id=doc_id,
            text=text,
            score=_distance_to_score(dist, space),
            metadata=dict(meta or {}),
        )
        for doc_id, text, dist, meta in zip(
            res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0]
        )
    ]


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Build a VectorDBBenchmark that inserts 1,000 mock documents into ChromaDB,
# then runs 100 queries and measures: average query latency, 95th percentile
# latency, and results-per-second throughput.

_VOCAB = (
    "rag retrieval generation embedding vector database index query latency "
    "model prompt agent tool token context chunk metadata filter cosine "
    "similarity search cluster shard replica cache batch stream pipeline "
    "document corpus ranking rerank hybrid keyword semantic memory scale "
    "tenant schema upsert delete update persist"
).split()


@dataclass
class BenchmarkResult:
    n_docs: int
    n_queries: int
    results_returned: int
    avg_latency_ms: float
    p95_latency_ms: float
    queries_per_second: float
    results_per_second: float


class VectorDBBenchmark:
    """Insert n_docs mock documents into a fresh ChromaDB collection, then time
    n_queries queries. Query vectors are embedded up front, so the latencies
    measure ChromaDB's search only (not the embedding function)."""

    def __init__(
        self,
        embed_fn: EmbedFn = mock_embed,
        client: ClientAPI | None = None,
        n_docs: int = 1000,
        n_queries: int = 100,
        n_results: int = 5,
        batch_size: int = 500,
        seed: int = 42,
    ):
        _require_chromadb()
        self.embed_fn = embed_fn
        self.client = client or chromadb.EphemeralClient()
        self.n_docs = n_docs
        self.n_queries = n_queries
        self.n_results = n_results
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def _random_texts(self, n: int, words: int = 8) -> list[str]:
        return [" ".join(self._rng.choice(_VOCAB, size=words)) for _ in range(n)]

    def run(self) -> BenchmarkResult:
        collection = create_cosine_collection(self.client, _unique_name("benchmark"))
        try:
            # Insert in batches (ChromaDB caps the size of a single add() call).
            docs = self._random_texts(self.n_docs)
            for start in range(0, self.n_docs, self.batch_size):
                chunk = docs[start:start + self.batch_size]
                _add_records(
                    collection,
                    ids=[f"doc_{start + i}" for i in range(len(chunk))],
                    texts=chunk,
                    embeddings=self.embed_fn(chunk),
                )

            query_vectors = self.embed_fn(self._random_texts(self.n_queries, words=4))

            # Untimed warm-up so one-off setup cost doesn't skew the first sample.
            collection.query(query_embeddings=[query_vectors[0]], n_results=self.n_results)

            latencies: list[float] = []
            total_results = 0
            for q_vec in query_vectors:
                t0 = time.perf_counter()
                res = collection.query(query_embeddings=[q_vec], n_results=self.n_results)
                latencies.append(time.perf_counter() - t0)
                total_results += len(res["ids"][0])
        finally:
            self.client.delete_collection(collection.name)

        total_time = sum(latencies)
        return BenchmarkResult(
            n_docs=self.n_docs,
            n_queries=len(latencies),
            results_returned=total_results,
            avg_latency_ms=float(np.mean(latencies) * 1000),
            p95_latency_ms=float(np.percentile(latencies, 95) * 1000),
            queries_per_second=len(latencies) / total_time,
            results_per_second=total_results / total_time,
        )


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Implement a MultiCollectionSearch class for ChromaDB that queries across
# multiple named collections and merges results by score, returning a unified
# ranked list.

class MultiCollectionSearch:
    """Query several named collections with one query and merge the hits into a
    single list ranked by score (higher = more similar). Distances are converted
    to scores per collection, so cosine and l2 collections can be mixed."""

    def __init__(self, client: ClientAPI | None = None, embed_fn: EmbedFn = mock_embed):
        _require_chromadb()
        self.client = client or chromadb.EphemeralClient()
        self.embed_fn = embed_fn
        self._names: list[str] = []

    def add(
        self,
        collection_name: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Embed and store documents in a (cosine) collection, creating it if needed."""
        collection = create_cosine_collection(self.client, collection_name)
        _add_records(collection, ids, texts, self.embed_fn(texts), metadatas)
        if collection_name not in self._names:
            self._names.append(collection_name)

    def search(
        self,
        query: str,
        collection_names: list[str] | None = None,
        k: int = 5,
        per_collection_k: int | None = None,
    ) -> list[SearchHit]:
        """Top-k hits across `collection_names` (default: every collection added
        through this object), merged and sorted by descending score."""
        names = collection_names if collection_names is not None else self._names
        per_k = per_collection_k or k
        q_vec = self.embed_fn([query])[0]

        hits: list[SearchHit] = []
        for name in names:
            collection = self.client.get_collection(name=name)
            hits.extend(_query_hits(collection, q_vec, per_k))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Design and implement a tenant-isolated vector store using ChromaDB: each
# tenant gets their own collection, and a TenantVectorStore class routes
# add/search operations to the correct collection by tenant ID.

_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")


class TenantVectorStore:
    """One ChromaDB collection per tenant. Every operation takes a tenant_id and is
    routed to that tenant's collection only, so a search can never return another
    tenant's documents. Tenant ids are validated (not sanitised) so two different
    ids can never map onto the same collection name."""

    def __init__(
        self,
        embed_fn: EmbedFn = mock_embed,
        client: ClientAPI | None = None,
        prefix: str = "tenant_",
    ):
        _require_chromadb()
        self.embed_fn = embed_fn
        self.client = client or chromadb.EphemeralClient()
        self.prefix = prefix

    def _collection_name(self, tenant_id: str) -> str:
        if not _TENANT_ID_RE.match(tenant_id):
            raise ValueError(
                f"Invalid tenant id {tenant_id!r}: use 1-63 letters, digits, '_' or '-', "
                "starting with a letter or digit."
            )
        return f"{self.prefix}{tenant_id}"

    def _get(self, tenant_id: str) -> Collection | None:
        """The tenant's collection, or None if the tenant has no data yet."""
        name = self._collection_name(tenant_id)
        try:
            return self.client.get_collection(name=name)
        except Exception:   # the "not found" exception type differs between chromadb versions
            return None

    def add_many(
        self,
        tenant_id: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        collection = create_cosine_collection(self.client, self._collection_name(tenant_id))
        metadatas = metadatas or [{} for _ in ids]
        # Stamp every record with its tenant (also guarantees non-empty metadata).
        stamped = [{**(m or {}), "tenant_id": tenant_id} for m in metadatas]
        _add_records(collection, ids, texts, self.embed_fn(texts), stamped, upsert=True)

    def add(self, tenant_id: str, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.add_many(tenant_id, [doc_id], [text], [metadata or {}])

    def search(
        self,
        tenant_id: str,
        query: str,
        k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Top-k hits from this tenant's collection only. Unknown tenant -> []."""
        collection = self._get(tenant_id)
        if collection is None:
            return []
        return _query_hits(collection, self.embed_fn([query])[0], k, where)

    def count(self, tenant_id: str) -> int:
        collection = self._get(tenant_id)
        return 0 if collection is None else collection.count()

    def delete(self, tenant_id: str, doc_ids: list[str]) -> None:
        collection = self._get(tenant_id)
        if collection is not None:
            collection.delete(ids=doc_ids)

    def delete_tenant(self, tenant_id: str) -> bool:
        """Drop the tenant's whole collection. Returns True if it existed."""
        if self._get(tenant_id) is None:
            return False
        self.client.delete_collection(self._collection_name(tenant_id))
        return True


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Write a script that migrates a ChromaDB collection to a JSON export format
# (list of {id, text, embedding, metadata} dicts) and another script that imports
# it back - useful for backups and migration between providers.

def export_collection(collection: Collection, path: str | Path, page_size: int = 500) -> int:
    """Dump every record of `collection` to a JSON file as a list of
    {id, text, embedding, metadata} dicts. Returns the number of records."""
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = collection.get(
            include=["documents", "embeddings", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids = page["ids"]
        if len(ids) == 0:
            break
        for i, doc_id in enumerate(ids):
            records.append({
                "id": doc_id,
                "text": page["documents"][i],
                "embedding": [float(x) for x in page["embeddings"][i]],
                "metadata": dict(page["metadatas"][i] or {}),
            })
        offset += len(ids)

    Path(path).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return len(records)


def import_collection(
    client: ClientAPI,
    name: str,
    path: str | Path,
    space: str = "cosine",
    batch_size: int = 500,
) -> Collection:
    """Load a JSON export into collection `name` (created if missing, with the given
    distance space). Uses upsert, so re-running the import is safe and never
    duplicates records. The stored embeddings are reused - nothing is re-embedded."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = {"id", "text", "embedding", "metadata"}
    for n, rec in enumerate(records):
        if not missing <= rec.keys():
            raise ValueError(f"Record #{n} is missing keys: {sorted(missing - rec.keys())}")

    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": space})
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        _add_records(
            collection,
            ids=[r["id"] for r in chunk],
            texts=[r["text"] for r in chunk],
            embeddings=[r["embedding"] for r in chunk],
            metadatas=[r["metadata"] for r in chunk],
            upsert=True,
        )
    return collection


if __name__ == "__main__":
    if chromadb is None:
        print("chromadb is not installed. Run: pip install chromadb")
        raise SystemExit(0)

    try:
        print("=== Exercise 1: VectorDBBenchmark (1,000 docs, 100 queries, offline mock) ===")
        r = VectorDBBenchmark().run()
        print(f"  docs={r.n_docs}  queries={r.n_queries}  results returned={r.results_returned}")
        print(f"  average latency : {r.avg_latency_ms:.3f} ms")
        print(f"  p95 latency     : {r.p95_latency_ms:.3f} ms")
        print(f"  throughput      : {r.queries_per_second:.1f} queries/s | {r.results_per_second:.1f} results/s")

        print("\n=== Exercise 2: MultiCollectionSearch (offline, mock embeddings) ===")
        multi = MultiCollectionSearch()
        multi.add(_unique_name("papers"), ["p1", "p2"], [
            "RAG retrieves documents and injects them into the LLM context",
            "Fine-tuning updates model weights on task specific data",
        ])
        multi.add(_unique_name("faq"), ["f1", "f2"], [
            "How do I use RAG with a vector database",
            "What is the pricing for the managed plan",
        ])
        for hit in multi.search("RAG vector database retrieval", k=3):
            print(f"  [{hit.score:.4f}] ({hit.collection}) {hit.id}: {hit.text}")

        print("\n=== Exercise 3: TenantVectorStore (offline, mock embeddings) ===")
        tenants = TenantVectorStore(prefix=_unique_name("demo") + "_")
        tenants.add("acme", "a1", "Acme internal roadmap for the vector search launch", {"dept": "eng"})
        tenants.add("acme", "a2", "Acme quarterly revenue forecast")
        tenants.add("globex", "g1", "Globex secret recipe for vector pipelines")
        print("  counts:", {t: tenants.count(t) for t in ("acme", "globex", "initech")})
        print("  acme search   :", [(h.id, round(h.score, 3)) for h in tenants.search("acme", "vector search")])
        print("  globex search :", [(h.id, round(h.score, 3)) for h in tenants.search("globex", "vector search")])
        print("  unknown tenant:", tenants.search("initech", "vector search"))
        print("  acme filtered (dept == eng):",
              [h.id for h in tenants.search("acme", "roadmap", where={"dept": {"$eq": "eng"}})])
        try:
            tenants.add("bad/tenant", "x", "nope")
        except ValueError as exc:
            print("  rejected:", exc)
        print("  delete_tenant('globex'):", tenants.delete_tenant("globex"), "| count now", tenants.count("globex"))

        print("\n=== Exercise 4: Export / import round trip (offline, mock embeddings) ===")
        client = chromadb.EphemeralClient()
        source = create_cosine_collection(client, _unique_name("source"))
        texts = [
            "RAG connects LLMs to external knowledge bases.",
            "Fine-tuning adapts model weights using task-specific data.",
            "Vector databases index embeddings for fast similarity search.",
        ]
        _add_records(
            source,
            ids=["doc_0", "doc_1", "doc_2"],
            texts=texts,
            embeddings=mock_embed(texts),
            metadatas=[{"category": "technique", "index": 0}, {"category": "technique", "index": 1}, None],
        )
        export_path = DATA_DIR / "chroma_export.json"
        n_exported = export_collection(source, export_path)
        print(f"  exported {n_exported} records -> {export_path}")

        restored = import_collection(client, _unique_name("restored"), export_path)
        original = source.get(include=["documents", "embeddings", "metadatas"])
        copy = restored.get(include=["documents", "embeddings", "metadatas"])
        order_a, order_b = np.argsort(original["ids"]), np.argsort(copy["ids"])
        same_ids = sorted(original["ids"]) == sorted(copy["ids"])
        same_text = [original["documents"][i] for i in order_a] == [copy["documents"][i] for i in order_b]
        same_vecs = np.allclose(np.asarray(original["embeddings"])[order_a], np.asarray(copy["embeddings"])[order_b])
        print(f"  restored {restored.count()} records | ids match: {same_ids} | "
              f"text match: {same_text} | embeddings match: {same_vecs}")

        probe = mock_embed(["vector similarity search"])[0]
        print("  same top hit in both:",
              _query_hits(source, probe, 1)[0].id == _query_hits(restored, probe, 1)[0].id)
    except Exception as exc:   # chromadb runtime errors
        print(f"Exercises failed: {type(exc).__name__}: {exc}")
