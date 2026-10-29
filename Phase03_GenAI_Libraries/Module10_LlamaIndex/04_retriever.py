# Module 10 - LlamaIndex
# 10.5 Retriever - Used Independently
# Sometimes you want retrieval without synthesis - for building custom pipelines.
# VectorIndexRetriever returns the top-k nodes + scores and never calls the LLM.
#
# NOTE: with an API key in a .env file in this folder, building the index makes REAL, billed
# embedding calls (retrieval itself makes no chat call). With no key the __main__ demo runs on
# offline hashed bag-of-words embeddings instead.

try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.retrievers import VectorIndexRetriever
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

from _openrouter_client import configure_settings_or_offline


def build_documents() -> list[Document]:
    return [
        Document(text="ChromaDB is an open-source embedding database with a simple Python API.", metadata={"tag": "vector-db"}),
        Document(text="Pinecone is a managed vector database designed for production workloads.", metadata={"tag": "vector-db"}),
        Document(text="FAISS is a library by Meta for efficient similarity search on dense vectors.", metadata={"tag": "vector-search"}),
        Document(text="pgvector adds vector similarity search to PostgreSQL.", metadata={"tag": "vector-db"}),
        Document(text="Weaviate is an open-source vector database with a GraphQL API.", metadata={"tag": "vector-db"}),
    ]


if __name__ == "__main__":
    configure_settings_or_offline()

    index = VectorStoreIndex.from_documents(build_documents())
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

    # Retrieve without synthesising an answer
    nodes = retriever.retrieve("open source vector databases")

    print("Retrieved nodes:")
    for node in nodes:
        print(f" [{node.score:.4f}] {node.text}")
        print(f"          tag: {node.metadata.get('tag')}")
