# Module 10 - LlamaIndex
# 10.3 Documents and Nodes
# A Document wraps raw text + metadata. A node parser (SentenceSplitter) cuts documents into
# Nodes - the chunks that actually get embedded and retrieved. Each node keeps its source metadata.
#
# NOTE: needs `pip install llama-index`. This file needs NO API key and makes no model call.
# SentenceSplitter counts tokens with tiktoken, which may download its encoding file the first
# time it runs (a plain file download, not an API call).

try:
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

import _openrouter_client  # noqa: F401  (loads .env and forces UTF-8 stdout)


# ── Create documents from text ─────────────────────────────────────────────────

def build_documents() -> list[Document]:
    """The three sample documents from the book, each tagged with source/category metadata."""
    return [
        Document(
            text="""LlamaIndex is a data framework for LLM applications.
            It provides tools for ingesting, structuring, and querying data
            using large language models. The core abstraction is the Index,
            which organises documents into a structure optimised for retrieval.""",
            metadata={"source": "llamaindex_overview", "category": "framework"},
        ),
        Document(
            text="""Vector store indexes store document embeddings in a vector database.
            They support fast semantic search using approximate nearest-neighbour algorithms.
            LlamaIndex integrates with ChromaDB, Pinecone, Weaviate, and many other backends.""",
            metadata={"source": "vector_store_guide", "category": "retrieval"},
        ),
        Document(
            text="""The query engine is the highest-level abstraction in LlamaIndex.
            It takes a natural language question, retrieves relevant nodes,
            and uses an LLM to synthesise a coherent answer from the retrieved context.""",
            metadata={"source": "query_engine_docs", "category": "querying"},
        ),
    ]


# ── Parse documents into nodes (chunks) ────────────────────────────────────────

def parse_nodes(documents: list[Document], chunk_size: int = 128, chunk_overlap: int = 20) -> list:
    """Split documents into nodes. Metadata is copied onto every node."""
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return parser.get_nodes_from_documents(documents)


if __name__ == "__main__":
    documents = build_documents()
    nodes = parse_nodes(documents)

    print(f"Documents: {len(documents)}, Nodes: {len(nodes)}")
    for n in nodes:
        print(f" Node {n.node_id[:8]}... | {len(n.text)} chars | source: {n.metadata.get('source')}")

    # A TextNode can also be built by hand when you already have your own chunks.
    manual = TextNode(text="A hand-made chunk.", metadata={"source": "manual"})
    print(f"\nManual node: {manual.node_id[:8]}... | {len(manual.text)} chars | source: {manual.metadata.get('source')}")
