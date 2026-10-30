# Module 11 - Vector Databases
# 11.2 ChromaDB
# ChromaDB is the easiest vector database to get started with. It runs
# in-process or as a server, and persists data to disk. This file walks through
# a persistent client, adding documents with pre-computed embeddings, querying,
# metadata filtering, updating and deleting.
#
# NOTE: Needs the `chromadb` package (pip install chromadb). Embeddings use
# OPENAI_API_KEY (or OPENROUTER_API_KEY) in a .env file in
# this folder - REAL, billed API calls. With no key set, the demo falls back to
# an offline mock embedder, so it still runs (results are then not semantic).
# Data is persisted to ./data/chroma_db_<openai|mock>/ in this folder.
#
# NOTE: Targets the chromadb 1.x API as used in the book. The book sets cosine
# distance through metadata={"hnsw:space": "cosine"}; newer releases may prefer
# a collection `configuration=` argument, but the metadata form is kept here.

from __future__ import annotations

from typing import TYPE_CHECKING

from _openrouter_client import DATA_DIR, get_embed_fn

try:
    import chromadb
except ImportError:   # handled gracefully in __main__
    chromadb = None

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

# ── Embedding helper ───────────────────────────────────────────────────────────

embed, EMBED_LABEL = get_embed_fn()   # real embeddings if a key is set, else offline mock

COLLECTION_NAME = "ai_knowledge_base"

DOCUMENTS = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]


# ── Persistent client + collection ─────────────────────────────────────────────

def make_client(path: str | None = None) -> ClientAPI:
    """Persistent client (saves to disk)."""
    path = path or str(DATA_DIR / f"chroma_db_{EMBED_LABEL}")
    return chromadb.PersistentClient(path=path)


def reset_collection(client: ClientAPI, name: str = COLLECTION_NAME) -> Collection:
    """Get or create a collection with cosine similarity, emptied so the demo is
    repeatable on every run. Only touches the collection `name`."""
    existing = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    if existing.count() > 0:
        client.delete_collection(name)
        existing = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return existing


# ── Add documents with pre-computed embeddings ─────────────────────────────────

def add_documents(collection: Collection, documents: list[str]) -> None:
    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [{"category": "technique", "index": i} for i in range(len(documents))]
    embeddings = embed(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Collection size: {collection.count()}")


# ── Query ──────────────────────────────────────────────────────────────────────

def query_documents(collection: Collection, query: str, n_results: int = 3) -> None:
    q_embed = embed([query])[0]

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    print("\nTop results:")
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        print(f"  [{dist:.4f}] {doc}")
        print(f"               category: {meta['category']}")


# ── Metadata filtering ─────────────────────────────────────────────────────────

def query_filtered(collection: Collection, query: str, n_results: int = 3) -> None:
    """Filter by metadata before similarity ranking."""
    q_embed = embed([query])[0]

    results_filtered = collection.query(
        query_embeddings=[q_embed],
        n_results=n_results,
        where={"category": {"$eq": "technique"}},   # pre-filter
        include=["documents", "distances"],
    )

    print("\nFiltered results (category == 'technique'):")
    for doc, dist in zip(results_filtered["documents"][0], results_filtered["distances"][0]):
        print(f"  [{dist:.4f}] {doc}")


# ── Update and delete ──────────────────────────────────────────────────────────

def update_document(collection: Collection, doc_id: str, new_text: str) -> None:
    """Replace a document's text and re-embed it."""
    collection.update(
        ids=[doc_id],
        documents=[new_text],
        embeddings=embed([new_text]),
    )


def delete_document(collection: Collection, doc_id: str) -> None:
    collection.delete(ids=[doc_id])
    print(f"After delete, count: {collection.count()}")


if __name__ == "__main__":
    if chromadb is None:
        print("chromadb is not installed. Run: pip install chromadb")
        raise SystemExit(0)

    if EMBED_LABEL == "mock":
        print("No embedding API key found - using the offline mock embedder (results are not semantic).")

    try:
        client = make_client()
        collection = reset_collection(client)

        add_documents(collection, DOCUMENTS)

        query = "how do I add knowledge to an LLM without retraining?"
        query_documents(collection, query)
        query_filtered(collection, query)

        update_document(
            collection,
            "doc_0",
            "RAG retrieves external documents and injects them into the LLM context.",
        )
        print("\nUpdated doc_0:", collection.get(ids=["doc_0"])["documents"][0])

        delete_document(collection, "doc_4")
    except Exception as exc:   # embedding API / chromadb runtime errors
        print(f"ChromaDB demo failed: {type(exc).__name__}: {exc}")
