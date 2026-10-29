# Module 09 - LangChain
# 9.9 Module 09 Exercises
#
# Every exercise here takes an injectable `llm` (any LangChain Runnable that
# maps a rendered prompt to a message) and, where needed, a `retriever`. The
# defaults are an offline fake model and an in-memory keyword retriever, so the
# whole chain wiring runs without an API key. With a key, the __main__ block
# also runs exercises 1 and 3 against a real chat model, and exercises 2 and 4
# against a real Chroma retriever.
#
# NOTE: The offline part requires langchain-core and langchain-text-splitters.
# The real-model part makes REAL, billed API calls and needs a key in a .env
# file in this folder (plus langchain-openai / langchain-anthropic, and for the
# retriever also chromadb and langchain-community or langchain-chroma).

import hashlib
import re
from typing import Any

from _openrouter_client import (
    ensure_ready,
    get_chat_model,
    get_embeddings,
    is_ready,
    make_fake_llm,
)

# Print a clear message (instead of a traceback) when a library is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core", "langchain_text_splitters"]):
    raise SystemExit(0)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Shared helpers ─────────────────────────────────────────────────────────────

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_STOPWORDS = {"the", "what", "does", "for", "and", "how", "are", "with", "that", "this", "from"}


def parse_lines(text: str) -> list[str]:
    """Split model output into non-empty lines, stripping bullets/numbering."""
    lines = (_BULLET_RE.sub("", line).strip() for line in text.splitlines())
    return [line for line in lines if line]


def format_docs(documents: list[Document]) -> str:
    """Join retrieved documents into one context string."""
    return "\n\n".join(d.page_content for d in documents)


def _tokens(text: str) -> set[str]:
    """Lower-cased content words (naive plural stripping) for offline matching."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        w[:-1] if len(w) > 3 and w.endswith("s") else w
        for w in words
        if len(w) >= 3 and w not in _STOPWORDS
    }


def make_keyword_retriever(documents: list[Document], k: int = 3) -> Runnable:
    """Offline retriever: ranks documents by word overlap with the query and
    returns the top k that share at least one word. Same str -> list[Document]
    contract as vectorstore.as_retriever()."""
    def search(query: str) -> list[Document]:
        query_tokens = _tokens(query)
        scored = [(len(query_tokens & _tokens(d.page_content)), i) for i, d in enumerate(documents)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [documents[i] for _, i in scored[:k]]

    return RunnableLambda(search)


def _import_chroma():
    """Chroma lives in langchain_community in the guide; newer releases moved it
    to the dedicated langchain-chroma package, so fall back to that."""
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain_chroma import Chroma
    return Chroma


def build_chroma_retriever(documents: list[Document], embeddings: Any, k: int = 3) -> Runnable:
    """Real retriever: split, embed, store in an in-memory Chroma collection."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(documents)
    vectorstore = _import_chroma().from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ── Prompts (shared by the exercises and by the offline fake model) ────────────

FACTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You extract key facts from documents. Extract exactly 3 key facts from "
               "the document. Return one fact per line, with no numbering or bullets."),
    ("human", "{text}"),
])

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You write quiz questions. Write one quiz question that tests knowledge "
               "of the given fact. Return only the question."),
    ("human", "{fact}"),
])

PARAPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Generate {n} paraphrased versions of the user's search query. "
               "Return one per line, with no numbering or extra text."),
    ("human", "{query}"),
])

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following passage in at most 2 sentences."),
    ("human", "{text}"),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "The following are summaries of consecutive parts of one long document. "
               "Combine them into a single coherent summary of at most 3 sentences."),
    ("human", "{summaries}"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict fact-checker. Rate how well the ANSWER is grounded in the "
               "CONTEXT on a scale from 0 (not grounded at all) to 10 (fully supported by the "
               "context). Reply with the number only."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
])

NO_INFO_ANSWER = "I don't have that information."


# ── Offline fake model ─────────────────────────────────────────────────────────

def _mock_respond(prompt_text: str) -> str:
    """Deterministic stand-in for an LLM. It recognises each exercise's prompt by
    a phrase in the system message and answers with simple string logic. Not
    intelligent - just enough to exercise every step of the chains offline."""
    system, _, human = prompt_text.partition("\nHuman: ")
    system_lower = system.lower()

    if "fact-checker" in system_lower:
        context, _, answer = human.removeprefix("Context:\n").partition("\n\nAnswer:\n")
        if "don't have that information" in answer:
            return "10"
        answer_tokens = _tokens(answer)
        if not answer_tokens:
            return "0"
        return str(round(10 * len(answer_tokens & _tokens(context)) / len(answer_tokens)))

    if "only the provided context" in system_lower:
        context = system.partition("Context:\n")[2]
        question_tokens = _tokens(human)
        best = max(
            (c for c in context.split("\n\n") if c.strip()),
            key=lambda c: len(question_tokens & _tokens(c)),
            default="",
        )
        return best if question_tokens & _tokens(best) else NO_INFO_ANSWER

    if "paraphrased" in system_lower:
        q = human.strip().rstrip("?")
        return f"Explain {q}\n{q} - overview\nWhat should I know about: {q}"

    if "key facts" in system_lower:
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(human.split()))
        return "\n".join([s for s in sentences if s][:3])

    if "quiz question" in system_lower:
        return f"True or false: {human.strip().rstrip('.')}?"

    if "summar" in system_lower:
        lines = [" ".join(line.split()[:12]) for line in human.splitlines() if line.strip()]
        return " ".join(lines)[:150]

    return "(fake model) no canned reply for this prompt"


MOCK_LLM: Runnable = make_fake_llm(_mock_respond)


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Build an LCEL chain that: (a) takes a raw document text, (b) extracts 3 key
# facts using one LLM call, (c) generates a quiz question for each fact using a
# second call, and (d) returns the questions as a Python list.

def build_quiz_chain(llm: Runnable = MOCK_LLM) -> Runnable:
    """str (raw document text) -> list[str] (one quiz question per extracted fact)."""
    parser = StrOutputParser()

    # (a) + (b): document text -> list of 3 facts (first LLM call)
    facts_chain = (
        {"text": RunnablePassthrough()}
        | FACTS_PROMPT
        | llm
        | parser
        | RunnableLambda(lambda text: parse_lines(text)[:3])
    )

    # (c): one second-call question per fact
    question_chain = QUESTION_PROMPT | llm | parser

    def make_questions(facts: list[str]) -> list[str]:
        return [q.strip() for q in question_chain.batch([{"fact": f} for f in facts])]

    # (d): the chain returns a plain Python list of questions
    return facts_chain | RunnableLambda(make_questions)


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Implement a MultiQueryRetriever manually: for a given query, generate 3
# paraphrased versions of it, retrieve documents for each, deduplicate by
# content hash, and return the union.

def content_hash(doc: Document) -> str:
    """SHA-256 of the document text - identical content hashes identically,
    whatever its metadata."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


class MultiQueryRetriever:
    """Retrieve with several paraphrases of a query and return the de-duplicated
    union (first occurrence wins, original order kept).

    `retriever` is any Runnable mapping a query string to list[Document]."""

    def __init__(
        self,
        retriever: Runnable,
        llm: Runnable = MOCK_LLM,
        n_queries: int = 3,
        include_original: bool = False,
    ):
        self.retriever = retriever
        self.n_queries = n_queries
        self.include_original = include_original
        self._paraphrase_chain = PARAPHRASE_PROMPT | llm | StrOutputParser()

    def generate_queries(self, query: str) -> list[str]:
        """Ask the LLM for n paraphrases of the query."""
        raw = self._paraphrase_chain.invoke({"query": query, "n": self.n_queries})
        queries = parse_lines(raw)[: self.n_queries]
        if self.include_original:
            queries.insert(0, query)
        return queries

    def invoke(self, query: str) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for q in self.generate_queries(query):
            for doc in self.retriever.invoke(q):
                digest = content_hash(doc)
                if digest not in seen:
                    seen.add(digest)
                    unique.append(doc)
        return unique


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Build a summarisation chain that handles documents longer than the model's
# context window by splitting into chunks, summarising each chunk, then
# summarising the summaries (map-reduce pattern).

class MapReduceSummariser:
    """Map: summarise every chunk. Reduce: summarise the summaries - repeating in
    rounds if the summaries together still exceed max_context_chars - until one
    summary is left. `llm_calls` counts model calls for the last summarise()."""

    def __init__(
        self,
        llm: Runnable = MOCK_LLM,
        chunk_size: int = 3000,
        chunk_overlap: int = 200,
        max_context_chars: int = 6000,
    ):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.max_context_chars = max_context_chars
        parser = StrOutputParser()
        self.map_chain = MAP_PROMPT | llm | parser
        self.reduce_chain = REDUCE_PROMPT | llm | parser
        self.llm_calls = 0

    def _reduce_round(self, summaries: list[str]) -> list[str]:
        """Group neighbouring summaries into batches that fit the context budget
        (at least 2 per batch, so every round makes progress) and summarise each."""
        batches: list[list[str]] = []
        current: list[str] = []
        for summary in summaries:
            joined_len = sum(len(s) for s in current) + len(summary)
            if len(current) >= 2 and joined_len > self.max_context_chars:
                batches.append(current)
                current = []
            current.append(summary)
        batches.append(current)

        reduced: list[str] = []
        for batch in batches:
            if len(batch) == 1:       # lone leftover: nothing to combine
                reduced.append(batch[0])
                continue
            self.llm_calls += 1
            reduced.append(self.reduce_chain.invoke({"summaries": "\n\n".join(batch)}).strip())
        return reduced

    def summarise(self, text: str) -> str:
        self.llm_calls = 0
        chunks = self.splitter.split_text(text)
        if not chunks:
            return ""

        # Map step: one summary per chunk
        summaries = [s.strip() for s in self.map_chain.batch([{"text": c} for c in chunks])]
        self.llm_calls += len(chunks)

        # Reduce step(s): summarise the summaries until they fit one prompt
        while len(summaries) > 1 and sum(len(s) for s in summaries) > self.max_context_chars:
            summaries = self._reduce_round(summaries)

        if len(summaries) == 1:
            return summaries[0]
        self.llm_calls += 1
        return self.reduce_chain.invoke({"summaries": "\n\n".join(summaries)}).strip()


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Add a response-quality filter to the RAG chain from section 9.8: after the
# answer is generated, run a second LLM call that scores the answer's
# groundedness (0-10) against the retrieved context. Return both the answer
# and the score.

def parse_score(text: str) -> float:
    """First number in the model's reply, clamped to 0-10. Raises ValueError if
    the reply contains no number."""
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        raise ValueError(f"No groundedness score found in model reply: {text!r}")
    return max(0.0, min(10.0, float(match.group())))


def build_scored_rag_chain(
    retriever: Runnable,
    llm: Runnable = MOCK_LLM,
    min_score: float | None = None,
) -> Runnable:
    """The 9.8 RAG chain plus a second LLM call that rates groundedness.

    Input: the question string. Output: dict with question, context, answer,
    score (0-10) and passed (True when score >= min_score, or always True when
    min_score is None) - the 'filter' decision is left to the caller."""
    parser = StrOutputParser()

    # Same retrieval + answer chain as section 9.8, but keeping context/question
    # in the output so the grader can see them.
    answer_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=RAG_PROMPT | llm | parser)
    )

    # Second LLM call: groundedness of the answer against the retrieved context
    grader = GROUNDEDNESS_PROMPT | llm | parser | RunnableLambda(parse_score)

    return (
        answer_chain
        | RunnablePassthrough.assign(score=grader)
        | RunnablePassthrough.assign(
            passed=RunnableLambda(lambda d: min_score is None or d["score"] >= min_score)
        )
    )


# ── Demo data ──────────────────────────────────────────────────────────────────

DOCUMENT_TEXT = (
    "LangChain is a framework for composing LLM-powered applications from reusable "
    "building blocks. LCEL uses the pipe operator to connect prompts, models and "
    "parsers into chains. Every runnable supports invoke, stream and batch. "
    "RunnableWithMessageHistory stores conversation history per session."
)

KB_DOCS = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."),
    Document(page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."),
    Document(page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."),
    Document(page_content="Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings for fast similarity search."),
    Document(page_content="Fine-tuning bakes knowledge into model weights. RAG retrieves knowledge at inference time. Both have trade-offs."),
]

LONG_TEXT = "\n\n".join([
    "LangChain started as a small library for chaining LLM calls together. It grew quickly "
    "as developers wanted reusable prompts, model wrappers and output parsers in one place.",
    "LCEL, the LangChain Expression Language, composes runnables with the pipe operator. "
    "Prompts, models, parsers and retrievers all share the same invoke, stream and batch methods.",
    "Retrieval-Augmented Generation connects a model to a knowledge base. Documents are split "
    "into chunks, embedded, stored in a vector database and retrieved at question time.",
    "Conversational memory keeps earlier turns available to the model. In LangChain this is "
    "handled by wrapping a chain with a message-history store keyed by session id.",
    "Long documents do not fit in one context window. The map-reduce pattern summarises each "
    "chunk separately and then summarises the summaries into a final answer.",
    "Evaluation closes the loop. Scoring answers for groundedness against retrieved context "
    "helps catch hallucinations before they reach users.",
])


def run_offline_demos() -> None:
    print("=== Exercise 1: quiz chain (offline, fake model) ===")
    for i, question in enumerate(build_quiz_chain().invoke(DOCUMENT_TEXT), start=1):
        print(f"  Q{i}: {question}")

    print("\n=== Exercise 2: MultiQueryRetriever (offline, keyword retriever) ===")
    mq = MultiQueryRetriever(make_keyword_retriever(KB_DOCS, k=2))
    query = "How does RAG retrieve documents?"
    print("  Paraphrases:", mq.generate_queries(query))
    results = mq.invoke(query)
    print(f"  Unique documents retrieved: {len(results)}")
    for doc in results:
        print(f"    - {doc.page_content[:70]}...")

    print("\n=== Exercise 3: map-reduce summariser (offline, fake model) ===")
    summariser = MapReduceSummariser(chunk_size=250, chunk_overlap=30, max_context_chars=300)
    summary = summariser.summarise(LONG_TEXT)
    print(f"  Input: {len(LONG_TEXT)} chars -> summary: {len(summary)} chars "
          f"({summariser.llm_calls} model calls)")
    print(f"  Summary: {summary}")

    print("\n=== Exercise 4: RAG + groundedness score (offline, fake model) ===")
    rag = build_scored_rag_chain(make_keyword_retriever(KB_DOCS), min_score=6)
    for q in ["What does RAG stand for?", "What is the capital of France?"]:
        result = rag.invoke(q)
        print(f"  Q: {q}\n  A: {result['answer']}\n  groundedness={result['score']:.0f}/10 "
              f"passed={result['passed']}\n")


def run_real_demos() -> None:
    llm = get_chat_model("auto", temperature=0.0)

    print("=== Exercise 1 (real model) ===")
    for i, question in enumerate(build_quiz_chain(llm).invoke(DOCUMENT_TEXT), start=1):
        print(f"  Q{i}: {question}")

    print("\n=== Exercise 3 (real model) ===")
    summariser = MapReduceSummariser(llm, chunk_size=250, chunk_overlap=30, max_context_chars=600)
    print(f"  {summariser.summarise(LONG_TEXT)}")
    print(f"  ({summariser.llm_calls} model calls)")

    if is_ready(["chromadb", "langchain_community|langchain_chroma"], embeddings=True):
        retriever = build_chroma_retriever(KB_DOCS, get_embeddings())

        print("\n=== Exercise 2 (real model + Chroma) ===")
        results = MultiQueryRetriever(retriever, llm).invoke("How does RAG retrieve documents?")
        for doc in results:
            print(f"    - {doc.page_content[:70]}...")

        print("\n=== Exercise 4 (real model + Chroma) ===")
        rag = build_scored_rag_chain(retriever, llm, min_score=6)
        for q in ["What does RAG stand for?", "What is the capital of France?"]:
            result = rag.invoke(q)
            print(f"  Q: {q}\n  A: {result['answer']}\n  groundedness={result['score']:.0f}/10 "
                  f"passed={result['passed']}\n")
    else:
        print("\nExercises 2 and 4 skipped: they need embeddings access plus chromadb "
              "and langchain-community (or langchain-chroma).")


if __name__ == "__main__":
    run_offline_demos()

    print("\n" + "=" * 60)
    if is_ready(llm="auto"):
        run_real_demos()
    else:
        print("Real-model demos skipped:")
        ensure_ready(llm="auto")
