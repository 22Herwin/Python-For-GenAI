# Module 11 - Vector Databases
# 11.4 Weaviate
# Weaviate is an open-source vector database with built-in hybrid search
# (vector + BM25 keyword), a GraphQL API, and a Python client. This file creates
# a collection with self-supplied embeddings, inserts objects, then runs vector
# search, hybrid search and a metadata-filtered search.
#
# NOTE: Needs the `weaviate-client` package (pip install weaviate-client) AND a
# running Weaviate instance - this script cannot run without one:
#   Local: docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
#   Cloud: set WEAVIATE_URL and WEAVIATE_API_KEY in a .env file in this folder.
# Embeddings use OPENAI_API_KEY (or OPENROUTER_API_KEY / OMNIROUTE_API_KEY) -
# REAL, billed API calls; with no key set, an offline mock embedder is used
# (results are then not semantic).
#
# NOTE: Written against the weaviate-client v4 API as used in the book
# (weaviate.auth.AuthApiKey, Configure.Vectorizer.none()). Newer v4 releases
# deprecate some of these names in favour of weaviate.classes.init.Auth and
# Configure.Vectors.self_provided(); update them if your version warns.

import os

from _openrouter_client import get_embed_fn

try:
    import weaviate
    import weaviate.classes as wvc
    from weaviate.classes.query import Filter
except ImportError:   # handled gracefully in __main__
    weaviate = None
    wvc = None
    Filter = None

# ── Embedding helper ───────────────────────────────────────────────────────────

embed, EMBED_LABEL = get_embed_fn()

COLLECTION_NAME = "AIDocument"

DOCUMENTS = [
    {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
    {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
    {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
]


# ── Connect (local Docker or Weaviate Cloud) ───────────────────────────────────

def connect():
    """Weaviate Cloud when WEAVIATE_URL + WEAVIATE_API_KEY are set, otherwise a local instance."""
    if os.getenv("WEAVIATE_URL") and os.getenv("WEAVIATE_API_KEY"):
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
        )
    return weaviate.connect_to_local()


# ── Create collection (schema) ─────────────────────────────────────────────────

def create_collection(client, name: str = COLLECTION_NAME) -> bool:
    """Create the collection if missing. Returns True if it was newly created."""
    if client.collections.exists(name):
        return False
    client.collections.create(
        name=name,
        properties=[
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),   # supply own embeddings
    )
    return True


# ── Insert objects with vectors ────────────────────────────────────────────────

def insert_documents(client, documents: list[dict], name: str = COLLECTION_NAME) -> None:
    collection = client.collections.get(name)
    vectors = embed([doc["text"] for doc in documents])
    with collection.batch.dynamic() as batch:
        for doc, vec in zip(documents, vectors):
            batch.add_object(
                properties={"text": doc["text"], "source": "guide", "category": "ai"},
                vector=vec,
            )
    failed = collection.batch.failed_objects
    if failed:
        print(f"{len(failed)} object(s) failed to insert; first error: {failed[0].message}")


# ── Vector search ──────────────────────────────────────────────────────────────

def vector_search(client, query: str, name: str = COLLECTION_NAME) -> None:
    collection = client.collections.get(name)
    results = collection.query.near_vector(
        near_vector=embed([query])[0],
        limit=3,
        return_properties=["text", "source"],
    )
    print("\nVector search:")
    for obj in results.objects:
        print(f"  {obj.properties['text']}")


# ── Hybrid search (BM25 + vector) ──────────────────────────────────────────────

def hybrid_search(client, keyword_query: str, vector_query: str, name: str = COLLECTION_NAME) -> None:
    collection = client.collections.get(name)
    results = collection.query.hybrid(
        query=keyword_query,                 # BM25 keyword query
        vector=embed([vector_query])[0],     # vector query
        alpha=0.5,                           # 0 = pure BM25, 1 = pure vector
        limit=3,
    )
    print("\nHybrid search (alpha=0.5):")
    for obj in results.objects:
        print(f"  {obj.properties['text']}")


# ── Metadata filter ────────────────────────────────────────────────────────────

def filtered_search(client, query: str, name: str = COLLECTION_NAME) -> None:
    collection = client.collections.get(name)
    results = collection.query.near_vector(
        near_vector=embed([query])[0],
        limit=3,
        filters=Filter.by_property("category").equal("ai"),
    )
    print("\nFiltered search (category == 'ai'):")
    for obj in results.objects:
        print(f"  {obj.properties['text']}")


if __name__ == "__main__":
    if weaviate is None:
        print("weaviate-client is not installed. Run: pip install weaviate-client")
        print("Weaviate patterns are in this file - they also need a running Weaviate instance.")
        raise SystemExit(0)

    if EMBED_LABEL == "mock":
        print("No embedding API key found - using the offline mock embedder (results are not semantic).")

    try:
        client = connect()
    except Exception as exc:   # no server reachable / bad credentials
        print(f"Could not connect to Weaviate: {type(exc).__name__}: {exc}")
        print("Weaviate patterns shown in this file - requires a running Weaviate instance.")
        print("Start locally: docker run -d -p 8080:8080 -p 50051:50051 "
              "cr.weaviate.io/semitechnologies/weaviate:latest")
        raise SystemExit(0)

    try:
        if create_collection(client):
            insert_documents(client, DOCUMENTS)
        else:
            print(f"Collection {COLLECTION_NAME!r} already exists - reusing it (no objects inserted).")

        vector_search(client, "external knowledge for LLMs")
        hybrid_search(client, "external knowledge LLM", "external knowledge for LLMs")
        filtered_search(client, "vector database")
    except Exception as exc:   # schema / query / embedding API errors
        print(f"Weaviate demo failed: {type(exc).__name__}: {exc}")
    finally:
        client.close()
