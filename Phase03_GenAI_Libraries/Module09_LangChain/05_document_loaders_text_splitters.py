# Module 09 - LangChain
# 9.6 Document Loaders and Text Splitters
# A Document is page_content plus metadata. Loaders produce Documents from files
# (DirectoryLoader, PyPDFLoader, ...) and text splitters cut them into
# overlapping chunks that fit an embedding model. Metadata is copied onto every
# chunk so the source stays traceable.
#
# NOTE: No API key needed - this file only builds and splits in-memory
# Documents. It requires langchain-core and langchain-text-splitters.

from _openrouter_client import ensure_ready

# Print a clear message (instead of a traceback) when a library is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core", "langchain_text_splitters"]):
    raise SystemExit(0)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Simulated loading ──────────────────────────────────────────────────────────
# (in practice use DirectoryLoader, PyPDFLoader, etc.)

def load_raw_docs() -> list[Document]:
    """Stand-in for a document loader: two in-memory Documents with metadata."""
    return [
        Document(
            page_content=(
                "LangChain provides a standard interface for interacting with "
                "large language models and related tools. It simplifies the "
                "development of context-aware reasoning applications by combining "
                "prompts, models, memory, and retrieval in composable pipelines. "
                "LCEL makes these pipelines declarative and easy to debug."
            ),
            metadata={"source": "langchain_docs.txt", "page": 1},
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) is a pattern that combines "
                "a retriever with a generator. The retriever finds relevant "
                "documents from a knowledge base. Those documents are injected "
                "into the generator's context window. This allows models to "
                "answer questions about information they were never trained on."
            ),
            metadata={"source": "rag_guide.txt", "page": 1},
        ),
    ]


# ── Splitting ──────────────────────────────────────────────────────────────────

def build_splitter() -> RecursiveCharacterTextSplitter:
    """Split on paragraphs first, then lines, sentences, words, characters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


if __name__ == "__main__":
    raw_docs = load_raw_docs()

    # Split into chunks
    splitter = build_splitter()
    chunks = splitter.split_documents(raw_docs)

    print(f"Original docs: {len(raw_docs)}, Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk.page_content)} chars | source: {chunk.metadata['source']}")
