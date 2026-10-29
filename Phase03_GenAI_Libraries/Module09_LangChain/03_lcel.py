# Module 09 - LangChain
# 9.4 LCEL - LangChain Expression Language
# LCEL uses the | pipe operator to compose runnables. Every component - prompts,
# models, parsers, retrievers - implements the Runnable interface with invoke,
# stream, and batch. This file covers a basic chain, parallel branches
# (RunnableParallel) and conditional routing (RunnableLambda).
#
# NOTE: The offline demo at the top of __main__ uses a fake model and needs no
# key. The second half makes REAL, billed API calls and requires a valid
# OPENAI_API_KEY, ANTHROPIC_API_KEY or OPENROUTER_API_KEY in a .env file in
# this folder (plus langchain-openai or langchain-anthropic).

from _openrouter_client import ensure_ready, get_chat_model, is_ready, make_fake_llm

# Print a clear message (instead of a traceback) when a library is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core"]):
    raise SystemExit(0)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

SAMPLE_TEXT = (
    "LangChain is a framework for developing applications powered by large "
    "language models. It provides abstractions for prompts, models, memory, and "
    "retrieval so developers can compose complex AI workflows from simple "
    "building blocks without writing boilerplate."
)


# ── Basic chain: prompt | model | parser ───────────────────────────────────────

def build_summary_chain(llm: Runnable) -> Runnable:
    """prompt | model | parser - input: {"sentences": int, "text": str}."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise the following text in exactly {sentences} sentences."),
        ("human", "{text}"),
    ])
    return prompt | llm | StrOutputParser()


# ── Parallel branches ──────────────────────────────────────────────────────────

def build_parallel_chain(llm: Runnable) -> Runnable:
    """Run two chains in parallel on the same input and combine the results
    into one dict - input: {"text": str}, output: {"summary", "keywords"}."""
    summarise_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise this in one sentence."),
        ("human", "{text}"),
    ])
    keywords_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract 3 keywords from this text. Return them comma-separated."),
        ("human", "{text}"),
    ])
    parser = StrOutputParser()

    return RunnableParallel(
        summary=summarise_prompt | llm | parser,
        keywords=keywords_prompt | llm | parser,
    )


# ── Conditional routing ────────────────────────────────────────────────────────

CODE_KEYWORDS = ["write", "code", "implement", "example", "function"]


def route(input_dict: dict) -> str:
    """Route to code or prose chain based on the question."""
    q = input_dict["input"].lower()
    if any(w in q for w in CODE_KEYWORDS):
        return "code"
    return "prose"


def build_router_chain(llm: Runnable) -> Runnable:
    """Pick the code chain or the prose chain per question - input: {"input": str}."""
    parser = StrOutputParser()

    code_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python expert. Provide a complete, working code example."),
        ("human", "{input}"),
    ])
    prose_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clear technical writer. Explain conceptually."),
        ("human", "{input}"),
    ])

    code_chain = code_prompt | llm | parser
    prose_chain = prose_prompt | llm | parser

    def run_routed(input_dict: dict) -> str:
        choice = route(input_dict)
        if choice == "code":
            return code_chain.invoke(input_dict)
        return prose_chain.invoke(input_dict)

    return RunnableLambda(run_routed)


# ── Demos ──────────────────────────────────────────────────────────────────────

def run_demos(llm: Runnable) -> None:
    print("--- Basic chain ---")
    chain = build_summary_chain(llm)
    print(chain.invoke({"sentences": 2, "text": SAMPLE_TEXT}))

    print("\n--- Parallel chain ---")
    parallel_chain = build_parallel_chain(llm)
    result = parallel_chain.invoke({
        "text": "Retrieval-Augmented Generation (RAG) improves LLM answers "
                "by fetching relevant documents at inference time."
    })
    print("Summary: ", result["summary"])
    print("Keywords:", result["keywords"])

    print("\n--- Conditional routing ---")
    router_chain = build_router_chain(llm)
    print(router_chain.invoke({"input": "Write a function to chunk text with overlap."}))
    print(router_chain.invoke({"input": "What is attention in transformers?"}))


def _fake_respond(prompt_text: str) -> str:
    """Canned replies so the chain wiring can be shown without a real model."""
    if "Python expert" in prompt_text:
        return "(fake model) code chain answered"
    if "technical writer" in prompt_text:
        return "(fake model) prose chain answered"
    if "3 keywords" in prompt_text:
        return "(fake model) langchain, retrieval, prompts"
    if "one sentence" in prompt_text:
        return "(fake model) one-sentence summary"
    return "(fake model) summary of the requested length"


if __name__ == "__main__":
    print("=== Offline demo (fake model, no API key) ===")
    run_demos(make_fake_llm(_fake_respond))

    print("\n=== Real model demo ===")
    if is_ready(llm="auto"):
        run_demos(get_chat_model("auto"))
    else:
        ensure_ready(llm="auto")
        print("Skipped the real-model demo.")
