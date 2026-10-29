# Module 09 - LangChain
# 9.7 Conversational Memory
# LangChain provides memory abstractions that maintain conversation history
# across turns in a structured way. RunnableWithMessageHistory wraps a chain,
# loads the session's history into the prompt's MessagesPlaceholder before each
# call, and appends the new human/AI messages afterwards.
#
# NOTE: The offline demo uses a fake model and needs no key. The second half
# makes REAL, billed API calls and requires a valid OPENAI_API_KEY,
# ANTHROPIC_API_KEY or OPENROUTER_API_KEY in a .env file in this folder.

from _openrouter_client import ensure_ready, get_chat_model, is_ready, make_fake_llm

# Print a clear message (instead of a traceback) when a library is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core"]):
    raise SystemExit(0)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

# Session store - keyed by session_id
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the history for a session, creating an empty one on first use."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chain_with_history(llm: Runnable) -> RunnableWithMessageHistory:
    """prompt | model | parser, wrapped so each session_id keeps its own history."""
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | parser

    # Wrap chain with history management
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def run_conversation(llm: Runnable, session_id: str) -> None:
    chain_with_history = build_chain_with_history(llm)
    config = {"configurable": {"session_id": session_id}}

    # Turn 1
    r1 = chain_with_history.invoke({"input": "What is RAG?"}, config=config)
    print("Turn 1:", r1[:100])

    # Turn 2 - model remembers turn 1
    r2 = chain_with_history.invoke(
        {"input": "What are the main components I just asked about?"}, config=config
    )
    print("Turn 2:", r2[:100])

    # Inspect stored history
    history = get_session_history(session_id)
    print(f"Messages in history: {len(history.messages)}")


def _fake_respond(prompt_text: str) -> str:
    """Reports how many human messages the prompt contained - proof that the
    stored history is being passed back in on every turn."""
    human_turns = prompt_text.count("\nHuman: ")
    return f"(fake model) this prompt contained {human_turns} human message(s)."


if __name__ == "__main__":
    print("=== Offline demo (fake model, no API key) ===")
    run_conversation(make_fake_llm(_fake_respond), session_id="offline-demo")

    print("\n=== Real model demo ===")
    if is_ready(llm="auto"):
        run_conversation(get_chat_model("auto"), session_id="user-123")
    else:
        ensure_ready(llm="auto")
        print("Skipped the real-model demo.")
