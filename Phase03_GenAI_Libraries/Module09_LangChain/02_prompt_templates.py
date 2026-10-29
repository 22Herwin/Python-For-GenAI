# Module 09 - LangChain
# 9.3 Prompt Templates
# ChatPromptTemplate turns variables into a list of chat messages. Use
# MessagesPlaceholder to splice a whole conversation history into the prompt.
#
# NOTE: No API key needed - this file only formats prompts and never calls a
# model. It requires the langchain-core package.

from _openrouter_client import ensure_ready

# Print a clear message (instead of a traceback) when a library is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core"]):
    raise SystemExit(0)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ── Simple template ────────────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])


# ── Template with conversation history placeholder ─────────────────────────────

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


if __name__ == "__main__":
    # Format to messages (for inspection)
    msgs = prompt.format_messages(
        role="Python tutor",
        language="English",
        question="What is a generator?",
    )
    print(msgs)

    formatted = chat_prompt.format_messages(
        context="User is learning about RAG",
        history=[
            HumanMessage(content="What is retrieval?"),
        ],
        input="How is it different from fine-tuning?",
    )
    print(f"Total messages in formatted prompt: {len(formatted)}")
