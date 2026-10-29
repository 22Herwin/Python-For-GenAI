# Module 10 - LlamaIndex
# 10.7 Persistent Storage
# LlamaIndex persists indexes to disk as JSON files. Loading a persisted index avoids
# re-embedding on every run.
#
# NOTE: with an API key in a .env file in this folder, building the index and querying make
# REAL, billed API calls (loading the saved index afterwards makes no embedding call). With no
# key the __main__ demo runs on offline mock models. The book persists to /tmp/llama_index_store;
# that path does not exist on Windows, so this file uses data/llama_index_store next to it.

from pathlib import Path

try:
    from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

from _openrouter_client import configure_settings_or_offline

PERSIST_DIR = Path(__file__).parent / "data" / "llama_index_store"


if __name__ == "__main__":
    configure_settings_or_offline()

    # Build and save index
    documents = [
        Document(text="LlamaIndex persists indexes to disk as JSON files."),
        Document(text="Loading a persisted index avoids re-embedding on every run."),
    ]

    index = VectorStoreIndex.from_documents(documents)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"Index saved to {PERSIST_DIR}")
    print("Files:", sorted(p.name for p in PERSIST_DIR.iterdir()))

    # Load index back - no re-embedding needed
    storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    loaded_index = load_index_from_storage(storage_context)

    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("How does LlamaIndex handle persistence?")
    print("After reload:", response)
