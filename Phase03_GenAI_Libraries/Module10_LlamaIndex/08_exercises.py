# Module 10 - LlamaIndex
# 10.9 Module 10 Exercises
#
# Every exercise here accepts injectable `llm` / `embed_model` (or a `score_fn`) that default
# to the offline mock models from _openrouter_client.py - a deterministic hashing embedding
# and an extractive "LLM" - so the core logic (folder Q&A, index comparison, metadata
# filtering, two-stage re-ranking) can be tested with no API key and no network. The
# __main__ demo switches to the real models automatically when OPENAI_API_KEY /
# OPENROUTER_API_KEY / OMNIROUTE_API_KEY is set in a .env file in this folder.
#
# NOTE: needs `pip install llama-index`. In real mode every exercise makes REAL, billed API
# calls (Exercise 4 makes one chat call per first-stage node, i.e. 10 per question).

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.schema import NodeWithScore
    from llama_index.core.vector_stores import (
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

from _openrouter_client import get_models_or_offline, get_offline_embed_model, get_offline_llm

DATA_DIR = Path(__file__).parent / "data"
DOCS_DIR = DATA_DIR / "exercise_docs"

ScoreFn = Callable[[str, str], float]


def _llm_or_offline(llm: LLM | None) -> LLM:
    return llm if llm is not None else get_offline_llm()


def _embed_or_offline(embed_model: BaseEmbedding | None) -> BaseEmbedding:
    return embed_model if embed_model is not None else get_offline_embed_model()


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Build a document Q&A system that reads from a folder of .txt files, indexes
# them with LlamaIndex, and exposes a ask(question: str) -> dict function
# returning both the answer and the source filenames.

SAMPLE_DOCS = {
    "rag_basics.txt": (
        "Retrieval-Augmented Generation (RAG) retrieves relevant passages from an external "
        "knowledge base and gives them to a language model as context. This lets the model "
        "answer questions about data it was never trained on."
    ),
    "vector_databases.txt": (
        "Vector databases store embeddings and answer nearest-neighbour queries quickly. "
        "HNSW and IVF are the most common approximate search algorithms. "
        "ChromaDB, Pinecone and Weaviate are popular choices."
    ),
    "fine_tuning.txt": (
        "Fine-tuning continues training a pre-trained model on a curated dataset so it performs "
        "better on one specific task. It changes the model weights, unlike prompt engineering."
    ),
    "prompt_engineering.txt": (
        "Prompt engineering designs the input given to a model to get better outputs without any "
        "training. Techniques include few-shot examples and chain-of-thought prompting."
    ),
}


def write_sample_docs(folder: Path = DOCS_DIR) -> None:
    """Create the sample .txt files, unless the folder already holds some."""
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("*.txt")):
        return
    for name, text in SAMPLE_DOCS.items():
        (folder / name).write_text(text, encoding="utf-8")


def load_text_documents(folder: Path) -> list[Document]:
    """Read every .txt file in `folder` into a Document tagged with its file_name.
    (SimpleDirectoryReader(input_dir=..., required_exts=[".txt"]) is the built-in
    alternative; reading by hand keeps the file_name metadata explicit.)"""
    return [
        Document(text=path.read_text(encoding="utf-8"), metadata={"file_name": path.name})
        for path in sorted(folder.glob("*.txt"))
    ]


class DocumentQA:
    """Folder of .txt files -> VectorStoreIndex -> ask() returning answer + source filenames."""

    def __init__(
        self,
        folder: Path | str,
        llm: LLM | None = None,
        embed_model: BaseEmbedding | None = None,
        similarity_top_k: int = 3,
    ):
        documents = load_text_documents(Path(folder))
        if not documents:
            raise ValueError(f"No .txt files found in {folder}")
        self.index = VectorStoreIndex.from_documents(documents, embed_model=_embed_or_offline(embed_model))
        self.query_engine = self.index.as_query_engine(llm=_llm_or_offline(llm), similarity_top_k=similarity_top_k)

    def ask(self, question: str) -> dict:
        """Return {"answer": str, "sources": [filenames of the retrieved nodes, de-duplicated]}."""
        response = self.query_engine.query(question)
        sources: list[str] = []
        for source_node in response.source_nodes:
            name = source_node.node.metadata.get("file_name")
            if name and name not in sources:
                sources.append(name)
        return {"answer": str(response), "sources": sources}


_default_qa: DocumentQA | None = None


def ask(question: str, qa: DocumentQA | None = None) -> dict:
    """Module-level ask(question) -> dict. Uses `qa` if given, otherwise a lazily built
    offline DocumentQA over data/exercise_docs."""
    global _default_qa
    if qa is None:
        if _default_qa is None:
            write_sample_docs()
            _default_qa = DocumentQA(DOCS_DIR)
        qa = _default_qa
    return qa.ask(question)


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Compare VectorStoreIndex and SummaryIndex on the same document set of 10
# paragraphs. Run 5 questions on each and compare answer quality for specific
# fact retrieval vs document-wide synthesis.

PARAGRAPHS = [
    "Retrieval-Augmented Generation (RAG) gives a language model access to external documents at query time.",
    "ChromaDB stores embeddings locally and is popular for prototypes; it was first released in 2022.",
    "Pinecone is a fully managed vector database, designed for production workloads with high availability.",
    "HNSW builds a layered graph of vectors and answers nearest-neighbour queries in logarithmic time.",
    "Chunking splits long documents into pieces; a typical chunk size is 512 tokens with an overlap of 64 tokens.",
    "text-embedding-3-small produces vectors with 1536 dimensions and costs 0.02 dollars per million tokens.",
    "Fine-tuning changes model weights, while RAG leaves the model unchanged and only changes its input.",
    "Evaluation of a retriever commonly uses recall at k and mean reciprocal rank (MRR).",
    "Re-ranking applies a second, more expensive model to reorder the first-stage candidates.",
    "Production RAG systems also need caching, monitoring, access control and regular index refreshes.",
]

# (question, kind, keywords a good answer should contain)
COMPARISON_QUESTIONS = [
    ("How many dimensions does text-embedding-3-small produce?", "fact", ["1536"]),
    ("What chunk size and overlap are typical?", "fact", ["512", "64"]),
    ("Which metrics are used to evaluate a retriever?", "fact", ["recall", "mrr"]),
    ("Summarise the whole document: what does a complete RAG system involve?", "synthesis",
     ["retriev", "chunk", "embedding", "production"]),
    ("List every database or index technology mentioned across the paragraphs.", "synthesis",
     ["chromadb", "pinecone", "hnsw"]),
]


@dataclass
class ComparisonRow:
    question: str
    kind: str
    vector_answer: str
    summary_answer: str
    vector_nodes: int
    summary_nodes: int
    vector_coverage: float
    summary_coverage: float


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer - a crude quality proxy."""
    if not keywords:
        return 0.0
    lowered = answer.lower()
    return sum(1 for k in keywords if k in lowered) / len(keywords)


def compare_indexes(
    paragraphs: list[str] = PARAGRAPHS,
    questions: list[tuple[str, str, list[str]]] = COMPARISON_QUESTIONS,
    llm: LLM | None = None,
    embed_model: BaseEmbedding | None = None,
    top_k: int = 3,
) -> list[ComparisonRow]:
    """Run every question on a VectorStoreIndex (top-k nodes) and a SummaryIndex (all nodes)."""
    llm = _llm_or_offline(llm)
    documents = [Document(text=p) for p in paragraphs]

    vector_engine = VectorStoreIndex.from_documents(
        documents, embed_model=_embed_or_offline(embed_model)
    ).as_query_engine(llm=llm, similarity_top_k=top_k)
    summary_engine = SummaryIndex.from_documents(documents).as_query_engine(llm=llm)

    rows = []
    for question, kind, keywords in questions:
        v_resp = vector_engine.query(question)
        s_resp = summary_engine.query(question)
        rows.append(ComparisonRow(
            question=question,
            kind=kind,
            vector_answer=str(v_resp),
            summary_answer=str(s_resp),
            vector_nodes=len(v_resp.source_nodes),
            summary_nodes=len(s_resp.source_nodes),
            vector_coverage=_keyword_coverage(str(v_resp), keywords),
            summary_coverage=_keyword_coverage(str(s_resp), keywords),
        ))
    return rows


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Implement a metadata-filtered retriever: given a corpus tagged with
# {"year": 2024, "topic": "embeddings"}, filter retrieval to only return nodes
# matching both a topic and year from the query metadata.

FILTER_CORPUS = [
    Document(text="Embeddings map text to dense vectors; 2024 models like text-embedding-3 shrink cost.",
             metadata={"year": 2024, "topic": "embeddings"}),
    Document(text="Matryoshka embeddings can be truncated to fewer dimensions with little quality loss.",
             metadata={"year": 2024, "topic": "embeddings"}),
    Document(text="Early embeddings such as word2vec captured word similarity but not sentence context.",
             metadata={"year": 2023, "topic": "embeddings"}),
    Document(text="RAG pipelines in 2024 add re-ranking and hybrid search on top of dense embeddings retrieval.",
             metadata={"year": 2024, "topic": "rag"}),
    Document(text="Early RAG systems simply stuffed the top retrieved chunks into the prompt.",
             metadata={"year": 2023, "topic": "rag"}),
    Document(text="Agents in 2025 use tool calling and planning loops built on language models.",
             metadata={"year": 2025, "topic": "agents"}),
]


def _matches(node: NodeWithScore, topic: str | None, year: int | None) -> bool:
    if topic is not None and node.metadata.get("topic") != topic:
        return False
    if year is not None and node.metadata.get("year") != year:
        return False
    return True


class MetadataFilteredRetriever:
    """VectorIndexRetriever that only returns nodes whose metadata matches the
    requested topic AND year (either may be omitted to filter on just one)."""

    def __init__(
        self,
        documents: list[Document] = FILTER_CORPUS,
        embed_model: BaseEmbedding | None = None,
        similarity_top_k: int = 3,
    ):
        self.index = VectorStoreIndex.from_documents(documents, embed_model=_embed_or_offline(embed_model))
        self.similarity_top_k = similarity_top_k

    def retrieve(self, query: str, topic: str | None = None, year: int | None = None) -> list[NodeWithScore]:
        clauses = []
        if topic is not None:
            clauses.append(MetadataFilter(key="topic", value=topic, operator=FilterOperator.EQ))
        if year is not None:
            clauses.append(MetadataFilter(key="year", value=year, operator=FilterOperator.EQ))
        filters = MetadataFilters(filters=clauses, condition=FilterCondition.AND) if clauses else None

        retriever = VectorIndexRetriever(
            index=self.index, similarity_top_k=self.similarity_top_k, filters=filters
        )
        nodes = retriever.retrieve(query)
        # Defence in depth: a vector store backend that ignores filters must not leak other nodes.
        return [n for n in nodes if _matches(n, topic, year)]


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Build a two-stage pipeline: first retrieve top-10 nodes with a
# VectorIndexRetriever, then re-rank them using a second LLM call that scores
# each node's relevance to the question (0-10). Return only the top-3 by LLM
# score.

RERANK_PROMPT = (
    "You are grading search results.\n"
    "Question: {question}\n"
    "Passage: {passage}\n\n"
    "Rate how relevant the passage is to answering the question, from 0 (irrelevant) "
    "to 10 (directly answers it). Reply with the number only."
)


def parse_score(reply: str) -> float:
    """First number in the LLM reply, clamped to [0, 10]; 0.0 if there is none."""
    match = re.search(r"\d+(?:\.\d+)?", reply)
    if match is None:
        return 0.0
    return max(0.0, min(10.0, float(match.group())))


def make_llm_scorer(llm: LLM) -> ScoreFn:
    """score_fn(question, passage) -> 0-10 using one LLM call per passage."""

    def score(question: str, passage: str) -> float:
        reply = llm.complete(RERANK_PROMPT.format(question=question, passage=passage))
        return parse_score(reply.text)

    return score


def lexical_relevance_score(question: str, passage: str) -> float:
    """Offline stand-in for the LLM scorer: 10 * fraction of question words found in the passage."""
    q_words = set(re.findall(r"\w+", question.lower()))
    p_words = set(re.findall(r"\w+", passage.lower()))
    if not q_words:
        return 0.0
    return 10.0 * len(q_words & p_words) / len(q_words)


class TwoStageRetriever:
    """Stage 1: VectorIndexRetriever top `first_stage_k`. Stage 2: score each node 0-10
    with `score_fn` (an LLM call in real mode) and keep the best `final_k`."""

    def __init__(
        self,
        index: VectorStoreIndex,
        score_fn: ScoreFn = lexical_relevance_score,
        first_stage_k: int = 10,
        final_k: int = 3,
    ):
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=first_stage_k)
        self.score_fn = score_fn
        self.final_k = final_k
        self.last_candidates: list[NodeWithScore] = []   # stage-1 output of the latest call

    def retrieve(self, question: str) -> list[NodeWithScore]:
        candidates = self.retriever.retrieve(question)
        self.last_candidates = candidates

        rescored = [
            (self.score_fn(question, n.node.get_content()), n.score or 0.0, n)
            for n in candidates
        ]
        # Highest LLM score first; the vector score breaks ties.
        rescored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [NodeWithScore(node=n.node, score=llm_score) for llm_score, _, n in rescored[: self.final_k]]


RERANK_CORPUS = [
    "ChromaDB is an open-source embedding database with a simple Python API.",
    "Pinecone is a managed vector database designed for production workloads.",
    "FAISS is a library by Meta for efficient similarity search on dense vectors.",
    "pgvector adds vector similarity search to PostgreSQL.",
    "Weaviate is an open-source vector database with a GraphQL API.",
    "HNSW is a graph algorithm that powers most modern vector search.",
    "IVF partitions vectors into clusters for fast approximate search.",
    "Product quantisation compresses vectors to reduce memory usage.",
    "Prompt engineering optimises the input to an LLM without any training.",
    "Fine-tuning trains a model on new data for a specific task.",
    "Chain-of-thought prompting asks a model to reason step by step.",
    "Agents use language models to plan tasks and call tools.",
]


if __name__ == "__main__":
    llm, embed_model, mode = get_models_or_offline()
    # Explicit injection below is the primary path; Settings is also set as a safety net.
    Settings.llm = llm
    Settings.embed_model = embed_model
    if mode == "offline":
        print("[offline] No API key set - exercises run on mock models "
              "(extractive answers, hashed bag-of-words embeddings, lexical re-rank scores).\n")
    else:
        print("[real] Using the configured API provider (billed calls).\n")

    print("=== Exercise 1: Document Q&A over a folder of .txt files ===")
    write_sample_docs()
    qa = DocumentQA(DOCS_DIR, llm=llm, embed_model=embed_model)
    for question in ["What are HNSW and IVF used for?", "How is fine-tuning different from prompt engineering?"]:
        result = ask(question, qa=qa)
        print(f"Q: {question}\n  answer : {result['answer']}\n  sources: {result['sources']}")

    print("\n=== Exercise 2: VectorStoreIndex vs SummaryIndex (10 paragraphs, 5 questions) ===")
    for row in compare_indexes(llm=llm, embed_model=embed_model):
        print(f"\n[{row.kind}] {row.question}")
        print(f"  vector  ({row.vector_nodes} nodes, keyword coverage {row.vector_coverage:.0%}): {row.vector_answer}")
        print(f"  summary ({row.summary_nodes} nodes, keyword coverage {row.summary_coverage:.0%}): {row.summary_answer}")
    print("\n(Keyword coverage is only a rough proxy - read the answers. Expect the vector index to do well on"
          " 'fact' questions and the summary index on 'synthesis' ones.)")

    print("\n=== Exercise 3: Metadata-filtered retriever (topic + year) ===")
    filtered = MetadataFilteredRetriever(embed_model=embed_model)
    for node in filtered.retrieve("how do embeddings work", topic="embeddings", year=2024):
        print(f"  [{node.score:.4f}] {node.metadata} | {node.text[:70]}")
    print("  -- same query, year only (2023):")
    for node in filtered.retrieve("how do embeddings work", year=2023):
        print(f"  [{node.score:.4f}] {node.metadata} | {node.text[:70]}")

    print("\n=== Exercise 4: Two-stage retrieve (top-10) + LLM re-rank (top-3) ===")
    rerank_index = VectorStoreIndex.from_documents(
        [Document(text=t) for t in RERANK_CORPUS], embed_model=embed_model
    )
    scorer = make_llm_scorer(llm) if mode == "real" else lexical_relevance_score
    two_stage = TwoStageRetriever(rerank_index, score_fn=scorer, first_stage_k=10, final_k=3)
    question = "Which vector databases are open source?"
    final_nodes = two_stage.retrieve(question)
    print(f"Question: {question}")
    print(f"Stage 1 candidates: {len(two_stage.last_candidates)}")
    for node in final_nodes:
        print(f"  [score {node.score:.1f}/10] {node.text}")
