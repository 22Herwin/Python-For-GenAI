# Module 10 - LlamaIndex
# 10.4 VectorStoreIndex
# The most commonly used index - builds a vector store from document nodes and supports
# semantic search. index.as_query_engine() retrieves the top-k nodes and has the LLM answer.
#
# NOTE: with an API key in a .env file in this folder, from_documents() and query() make REAL,
# billed API calls (embeddings + chat). With no key the __main__ demo runs on offline mock
# models (hashed bag-of-words embeddings + an extractive "LLM"), so answers are not generated.

try:
    from llama_index.core import Document, VectorStoreIndex
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

from _openrouter_client import configure_settings_or_offline


def build_documents() -> list[Document]:
    return [
        Document(text="RAG combines retrieval with generation to answer questions from external knowledge."),
        Document(text="Fine-tuning trains a model on new data to improve performance on a specific task."),
        Document(text="Prompt engineering optimises the input to an LLM to get better outputs without any training."),
        Document(text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."),
        Document(text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."),
    ]


if __name__ == "__main__":
    configure_settings_or_offline()
    documents = build_documents()

    # Build index - embeds all documents automatically
    index = VectorStoreIndex.from_documents(documents)

    # Query engine - retrieves + synthesises answer
    query_engine = index.as_query_engine(similarity_top_k=3)

    response = query_engine.query("What is the difference between RAG and fine-tuning?")
    print("Answer:", response)
    print("\nSource nodes:")
    for node in response.source_nodes:
        print(f" Score {node.score:.4f}: {node.text[:60]}...")
