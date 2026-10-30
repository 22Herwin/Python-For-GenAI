# Module 10 - LlamaIndex
# 10.1-10.2 What LlamaIndex Is, Setup and Settings
# LlamaIndex (formerly GPT Index) specialises in data ingestion and indexing for LLM apps.
# Core concepts: Node (a chunk of text + metadata, the atomic unit), Index (a structure built
# from nodes: VectorStoreIndex, SummaryIndex, KeywordIndex), Query Engine (question -> retrieve
# nodes -> synthesise answer), Retriever (the retrieval part, usable on its own).
#
# NOTE: needs `pip install llama-index`. With an OPENAI_API_KEY (or OPENROUTER_API_KEY)
# in a .env file in this folder, Settings points at the real models; with no
# key the __main__ demo falls back to the offline mock models. Building the Settings objects
# makes no API call.

try:
    from llama_index.core import Settings
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

from _openrouter_client import configure_settings_or_offline


# ── Global settings ────────────────────────────────────────────────────────────
# The book configures the models like this (applied to all indexes and query engines):
#
#   Settings.llm = OpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"],
#                         temperature=0.1, max_tokens=1024)
#   Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small",
#                                          api_key=os.environ["OPENAI_API_KEY"])
#   Settings.chunk_size = 512     # tokens per chunk
#   Settings.chunk_overlap = 64   # token overlap between chunks
#
# configure_settings_or_offline() does exactly that (the same classes and values), but also
# works through OpenRouter and falls back to offline mocks when there is no key.

def setup_settings() -> str:
    """Apply the book's global Settings. Returns "real" or "offline"."""
    return configure_settings_or_offline(
        temperature=0.1,
        max_tokens=1024,
        chunk_size=512,      # tokens per chunk
        chunk_overlap=64,    # token overlap between chunks
    )


if __name__ == "__main__":
    mode = setup_settings()
    print(f"\nMode            : {mode}")
    print(f"Settings.llm    : {type(Settings.llm).__name__} ({Settings.llm.metadata.model_name})")
    print(f"Settings.embed  : {type(Settings.embed_model).__name__} ({Settings.embed_model.model_name})")
    print(f"chunk_size      : {Settings.chunk_size}")
    print(f"chunk_overlap   : {Settings.chunk_overlap}")
