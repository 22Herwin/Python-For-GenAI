# Module 11 - Vector Databases
# 11.1 Why a Dedicated Vector Database
# The in-memory VectorStore from Phase 2 (Module 08) works for development, but
# production systems need persistence, scale, filtering, efficient updates and
# multi-tenancy. The three most widely used options are ChromaDB (simple,
# open-source), Pinecone (managed, production) and Weaviate (open-source,
# hybrid search). No API key or external library is needed for this file.

# ── Production requirements ────────────────────────────────────────────────────

REQUIREMENTS: list[tuple[str, str, str]] = [
    # (requirement, what it means, limitation of the Phase 2 in-memory VectorStore)
    ("Persistence",  "Data survives restarts",
     "Everything is lost when the process exits"),
    ("Scale",        "Millions of vectors without loading everything into RAM",
     "The whole (n, dim) matrix lives in RAM; brute-force scan on every query"),
    ("Filtering",    "Combine vector search with metadata conditions",
     "Only post-filtering in Python after the similarity scan"),
    ("Updates",      "Add, delete and update documents efficiently",
     "Matrix is rebuilt / re-stacked on every change"),
    ("Multi-tenancy", "Isolate collections per user or tenant",
     "One flat store; isolation must be hand-rolled"),
]

# ── The three most widely used options ─────────────────────────────────────────

OPTIONS: list[tuple[str, str]] = [
    ("ChromaDB", "simple, open-source - runs in-process or as a server, persists to disk"),
    ("Pinecone", "managed, production - fully hosted, no infrastructure to run"),
    ("Weaviate", "open-source, hybrid search (vector + BM25), GraphQL API, Python client"),
]


def print_overview() -> None:
    print("Why a dedicated vector database?\n")
    for name, meaning, limitation in REQUIREMENTS:
        print(f"  {name:<14} {meaning}")
        print(f"  {'':<14} in-memory store: {limitation}")
    print("\nMost widely used options:\n")
    for name, summary in OPTIONS:
        print(f"  {name:<10} {summary}")


if __name__ == "__main__":
    print_overview()
