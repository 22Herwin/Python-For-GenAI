# Module 09 - LangChain
# 9.8 Building a Full RAG Chain with LangChain
# Documents -> split -> embed -> Chroma vector store -> retriever -> prompt ->
# model -> parser, all composed with LCEL. The prompt tells the model to answer
# ONLY from the retrieved context, so out-of-scope questions get a refusal.
#
# NOTE: This script makes REAL, billed API calls (embeddings + chat) and needs
# an OpenAI-compatible key: OPENAI_API_KEY (or OPENROUTER_API_KEY /
# OMNIROUTE_API_KEY) in a .env file in this folder. It also requires
# langchain-core, langchain-text-splitters, langchain-openai, chromadb and
# either langchain-community (as in the guide) or langchain-chroma.

from _openrouter_client import ensure_ready, get_chat_model, get_embeddings

# Print a clear message (instead of a traceback) when a library or key is missing.
if __name__ == "__main__" and not ensure_ready(
    ["langchain_core", "langchain_text_splitters", "chromadb", "langchain_community|langchain_chroma"],
    llm="openai",
    embeddings=True,
):
    raise SystemExit(0)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _import_chroma():
    """The guide imports Chroma from langchain_community; newer releases moved
    it to the dedicated langchain-chroma package, so fall back to that."""
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain_chroma import Chroma
    return Chroma


# ── 1. Documents ───────────────────────────────────────────────────────────────

DOCS = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."),
    Document(page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."),
    Document(page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."),
    Document(page_content="Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings for fast similarity search."),
    Document(page_content="Fine-tuning bakes knowledge into model weights. RAG retrieves knowledge at inference time. Both have trade-offs."),
]

# ── 4. RAG prompt ──────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])


def format_docs(documents: list[Document]) -> str:
    """Join retrieved documents into one context string."""
    return "\n\n".join(d.page_content for d in documents)


def build_retriever(documents: list[Document], embeddings, k: int = 3):
    """Steps 2-3: split, embed, and store in Chroma (in-memory), then return a
    top-k retriever."""
    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    # 3. Embed and store in Chroma (in-memory)
    Chroma = _import_chroma()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_rag_chain(retriever, llm: Runnable) -> Runnable:
    """Step 5: retriever -> context string, question passed through, then
    prompt | model | parser. Input is the question string."""
    parser = StrOutputParser()

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | parser
    )


# ── 6. Ask questions ───────────────────────────────────────────────────────────

QUESTIONS = [
    "What does RAG stand for?",
    "How is RAG different from fine-tuning?",
    "What is the capital of France?",   # Not in context - should say so
]


if __name__ == "__main__":
    embeddings = get_embeddings()                       # text-embedding-3-small
    llm = get_chat_model("openai", temperature=0.0)     # gpt-4o

    retriever = build_retriever(DOCS, embeddings)
    rag_chain = build_rag_chain(retriever, llm)

    for q in QUESTIONS:
        print(f"Q: {q}")
        print(f"A: {rag_chain.invoke(q)}\n")
