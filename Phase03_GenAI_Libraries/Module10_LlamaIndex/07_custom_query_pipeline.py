# Module 10 - LlamaIndex
# 10.8 Custom Query Pipeline
# For advanced use cases, build the query pipeline explicitly: a retriever + a response
# synthesizer (with a chosen response mode) wired into a RetrieverQueryEngine.
#
# NOTE: with an API key in a .env file in this folder, this makes REAL, billed API calls
# (embeddings + several chat calls per query). With no key the __main__ demo runs on offline
# mock models, where the response mode changes the prompts but answers stay extractive.

try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
    from llama_index.core.retrievers import VectorIndexRetriever
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

from _openrouter_client import configure_settings_or_offline


def build_documents() -> list[Document]:
    return [
        Document(text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."),
        Document(text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."),
        Document(text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."),
        Document(text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."),
    ]


if __name__ == "__main__":
    configure_settings_or_offline()

    index = VectorStoreIndex.from_documents(build_documents())
    retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

    # Different response modes
    for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
        synthesiser = get_response_synthesizer(response_mode=mode)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesiser)
        resp = engine.query("What algorithms are used in vector search?")
        print(f"[{mode.value}] {str(resp)[:120]}\n")
