# Module 09 - LangChain
# 9.2 Chat Models
# LangChain wraps provider SDKs behind one chat-model interface. ChatOpenAI and
# ChatAnthropic accept the same message types and expose the same invoke /
# stream / batch methods, so swapping providers is a one-line change.
#
# NOTE: This script makes REAL, billed API calls. It requires langchain-core
# plus langchain-openai and/or langchain-anthropic, and a valid OPENAI_API_KEY,
# ANTHROPIC_API_KEY or OPENROUTER_API_KEY in a .env file in this folder.

from _openrouter_client import ensure_ready, get_chat_model

# Print a clear message (instead of a traceback) when a library or key is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core"], llm=("openai", "anthropic")):
    raise SystemExit(0)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


# ── Models ─────────────────────────────────────────────────────────────────────

def build_models() -> tuple[BaseChatModel | None, BaseChatModel | None]:
    """Return (OpenAI model, Anthropic model). Either is None when no matching
    key is configured. Same interface, same arguments - only the class differs."""
    # OpenAI model (gpt-4o)
    llm_oai = get_chat_model("openai", temperature=0.7, max_tokens=1024)
    # Anthropic model (claude-sonnet-4-5)
    llm_ant = get_chat_model("anthropic", temperature=0.7, max_tokens=1024)
    return llm_oai, llm_ant


# Both accept the same message types
MESSAGES = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]


# ── Invoke, stream, batch ──────────────────────────────────────────────────────

def demo_invoke(llm: BaseChatModel) -> None:
    """Invoke (blocking): returns one AIMessage."""
    response = llm.invoke(MESSAGES)
    print(response.content)


def demo_stream(llm: BaseChatModel) -> None:
    """Stream: yields message chunks as they are generated."""
    for chunk in llm.stream(MESSAGES):
        print(chunk.content, end="", flush=True)
    print()


def demo_batch(llm: BaseChatModel) -> None:
    """Batch - run multiple inputs in parallel."""
    responses = llm.batch([
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ])
    for r in responses:
        print(r.content[:80])


if __name__ == "__main__":
    llm_oai, llm_ant = build_models()
    # The guide demonstrates invoke/stream/batch on the Anthropic model; fall
    # back to the OpenAI one when that is the only key available.
    llm = llm_ant or llm_oai
    print(f"OpenAI model:    {type(llm_oai).__name__ if llm_oai else 'not configured'}")
    print(f"Anthropic model: {type(llm_ant).__name__ if llm_ant else 'not configured'}")

    print("\n=== invoke ===")
    demo_invoke(llm)
    print("\n=== stream ===")
    demo_stream(llm)
    print("\n=== batch ===")
    demo_batch(llm)
