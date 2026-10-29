# Module 10 - LlamaIndex
# 10.6 Summary Index
# A SummaryIndex is designed for question-answering over long documents where you need to
# synthesise across many chunks rather than retrieve the most similar ones.
#
# NOTE: with an API key in a .env file in this folder, query() makes REAL, billed chat calls
# (a SummaryIndex sends ALL nodes to the LLM, not just top-k, and builds no embeddings of its
# own). With no key the __main__ demo runs on an offline extractive mock LLM.

try:
    from llama_index.core import Document, SummaryIndex
except ImportError:
    raise SystemExit("llama-index is not installed. Run: pip install llama-index") from None

from _openrouter_client import configure_settings_or_offline


def build_sections() -> list[Document]:
    """Simulate a long document split into multiple sections."""
    return [
        Document(text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."),
        Document(text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."),
        Document(text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."),
        Document(text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."),
        Document(text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."),
    ]


if __name__ == "__main__":
    configure_settings_or_offline()

    index = SummaryIndex.from_documents(build_sections())
    query_engine = index.as_query_engine()

    # SummaryIndex reads ALL nodes, not just top-k - good for document-wide questions
    response = query_engine.query("Which model should I use for a coding project?")
    print(response)
    print(f"\nNodes sent to the LLM: {len(response.source_nodes)}")
