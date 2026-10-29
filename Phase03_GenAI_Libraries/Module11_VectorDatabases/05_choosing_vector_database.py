# Module 11 - Vector Databases
# 11.5 Choosing a Vector Database
# Decision table from the book. Decision rule: start with ChromaDB in
# development. When you go to production, choose Pinecone if you want zero
# infrastructure work, or Weaviate/Qdrant if you need hybrid search or want to
# self-host. No API key or external library is needed for this file.

# ── Decision table ─────────────────────────────────────────────────────────────

DECISION_TABLE: dict[str, str] = {
    "Local development, simple setup": "ChromaDB",
    "Production, managed, no ops": "Pinecone",
    "Hybrid BM25 + vector search": "Weaviate",
    "Already using PostgreSQL": "pgvector",
    "Maximum scale, self-hosted": "Qdrant or Weaviate",
}

DECISION_RULE = (
    "Start with ChromaDB in development. When you go to production, choose Pinecone "
    "if you want zero infrastructure work, or Weaviate/Qdrant if you need hybrid "
    "search or want to self-host."
)


def recommend(requirement: str) -> str:
    """Look up the best choice for a requirement from the decision table."""
    try:
        return DECISION_TABLE[requirement]
    except KeyError:
        raise KeyError(f"Unknown requirement {requirement!r}. Known: {list(DECISION_TABLE)}") from None


if __name__ == "__main__":
    width = max(len(req) for req in DECISION_TABLE)
    print(f"{'Requirement':<{width}}  Best choice")
    print(f"{'-' * width}  {'-' * 18}")
    for requirement, choice in DECISION_TABLE.items():
        print(f"{requirement:<{width}}  {choice}")
    print(f"\nDecision rule: {DECISION_RULE}")
