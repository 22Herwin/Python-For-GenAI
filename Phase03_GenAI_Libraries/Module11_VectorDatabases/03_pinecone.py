# Module 11 - Vector Databases
# 11.3 Pinecone
# Pinecone is a fully managed vector database - no infrastructure to run. It
# scales to billions of vectors and is the most common choice for production
# RAG systems. This file creates a serverless index, upserts vectors, queries
# and filters by metadata.
#
# NOTE: Needs the `pinecone` package (pip install pinecone), a valid
# PINECONE_API_KEY and an embedding key (OPENAI_API_KEY / OPENROUTER_API_KEY /
# OMNIROUTE_API_KEY) in a .env file in this folder. This script talks to the
# external Pinecone service: it CREATES a serverless index (cloud aws, region
# us-east-1) in your Pinecone project if it doesn't exist yet, and makes REAL
# embedding API calls. The index is never deleted automatically.

import os
import time

from _openrouter_client import EMBEDDING_DIM, has_openai_access, openai_embed

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:   # handled gracefully in __main__
    Pinecone = None
    ServerlessSpec = None

# ── Embedding helper ───────────────────────────────────────────────────────────

embed = openai_embed   # text-embedding-3-small -> 1536 dimensions

INDEX_NAME = "ai-knowledge-base"

DOCUMENTS = [
    {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
    {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
    {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
]


# ── Connect and create the index ───────────────────────────────────────────────

def get_or_create_index(pc: "Pinecone", index_name: str = INDEX_NAME):
    """Create the serverless index if it doesn't exist, then return a handle to it."""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,   # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


# ── Upsert vectors ─────────────────────────────────────────────────────────────

def upsert_documents(index, documents: list[dict]) -> None:
    values = embed([doc["text"] for doc in documents])   # one batched embedding call
    vectors = [
        {
            "id": doc["id"],
            "values": vec,
            "metadata": {"text": doc["text"], "source": "guide"},
        }
        for doc, vec in zip(documents, values)
    ]

    index.upsert(vectors=vectors)

    # Serverless indexes are eventually consistent: wait briefly until the
    # upserted vectors are visible before querying.
    for _ in range(10):
        stats = index.describe_index_stats()
        if stats["total_vector_count"] >= len(documents):
            break
        time.sleep(1.0)
    print(index.describe_index_stats())


# ── Query ──────────────────────────────────────────────────────────────────────

def query_index(index, query: str, top_k: int = 3) -> None:
    q_vector = embed([query])[0]
    results = index.query(
        vector=q_vector,
        top_k=top_k,
        include_metadata=True,
    )

    for match in results["matches"]:
        print(f"[{match['score']:.4f}] {match['metadata']['text']}")


# ── Metadata filter ────────────────────────────────────────────────────────────

def query_filtered(index, query: str, top_k: int = 3) -> None:
    q_vector = embed([query])[0]
    filtered = index.query(
        vector=q_vector,
        top_k=top_k,
        filter={"source": {"$eq": "guide"}},
        include_metadata=True,
    )

    print("\nFiltered results (source == 'guide'):")
    for match in filtered["matches"]:
        print(f"[{match['score']:.4f}] {match['metadata']['text']}")


if __name__ == "__main__":
    if Pinecone is None:
        print("pinecone is not installed. Run: pip install pinecone")
    elif not os.getenv("PINECONE_API_KEY"):
        print("PINECONE_API_KEY is not set. Add it to a .env file in this folder (see .env.example).")
    elif not has_openai_access():
        print("OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY is not set. "
              "An embedding key is required (the index dimension must match the embedding model).")
    else:
        try:
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            index = get_or_create_index(pc)

            upsert_documents(index, DOCUMENTS)
            query_index(index, "how to add external knowledge to an LLM?")
            query_filtered(index, "how to add external knowledge to an LLM?")
        except Exception as exc:   # network / auth / quota errors from Pinecone or the embedding API
            print(f"Pinecone demo failed: {type(exc).__name__}: {exc}")
