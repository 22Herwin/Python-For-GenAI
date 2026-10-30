# Module 13 - LangGraph & Agents
# Shared chat-model factory - lets every 0X_*.py file in this folder build a
# LangChain chat model from a real OPENAI_API_KEY, or through a single
# OPENROUTER_API_KEY instead, with no other code changes. If GEMINI_API_KEY and/or
# GROQ_API_KEY are also set, they back up OpenRouter: when it fails (e.g. its free
# daily quota is used up) the same call is retried on Gemini, then Groq.
# All of them speak the OpenAI-compatible chat API, so ChatOpenAI is enough
# (the book's code only uses ChatOpenAI).
#
# NOTE: get_chat_model() needs the langchain-openai package. Every demo in this
# module also runs offline with a scripted fake LLM, so a key is only needed
# for the optional "--real" paths.

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return. Force UTF-8 stdout so a print() of a
# model reply never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_USE_OPENROUTER = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

# Backups behind OpenRouter, each used only if its key is set, in this order.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Model used when OPENAI_API_KEY is set (the model the book uses).
DEFAULT_MODEL = "gpt-4o"

LANGGRAPH_INSTALL_HINT = "pip install langgraph langchain-core langchain-openai python-dotenv"
NO_KEY_HINT = "OPENAI_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder."


def has_llm_access() -> bool:
    """True if any supported API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


class _FallbackChatModel:
    """Tries each chat model in order until one answers. It mirrors the two calls this
    module makes on a model, invoke() and bind_tools(): LangChain's own
    RunnableWithFallbacks has no bind_tools(), so tools are bound on every model first."""

    def __init__(self, models: list):
        self.models = models

    def invoke(self, *args, **kwargs):
        return self.models[0].with_fallbacks(self.models[1:]).invoke(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        bound = [m.bind_tools(*args, **kwargs) for m in self.models]
        return bound[0].with_fallbacks(bound[1:])

    def __getattr__(self, name):   # anything else comes from the first (primary) model
        return getattr(self.models[0], name)


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    """LangChain ChatOpenAI routed through direct OpenAI or OpenRouter, with optional
    Gemini / Groq backups behind OpenRouter. Returns None when no key is configured.

    Raises ImportError (with an install hint) if langchain-openai is missing."""
    if not has_llm_access():
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(f"langchain-openai is not installed. Run: {LANGGRAPH_INSTALL_HINT}") from exc

    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=model or DEFAULT_MODEL, api_key=os.environ["OPENAI_API_KEY"],
                          temperature=temperature)

    # (model, key, base url) for each backup that has a key, in fallback order. Groq goes
    # before Gemini on purpose: an agent replays its earlier tool calls in the next request,
    # and Gemini 3.x rejects that over the OpenAI-compatible API (it wants a "thought_signature"
    # the client never sends), so Gemini can only back up single-turn calls.
    backups = []
    if os.getenv("GROQ_API_KEY"):
        backups.append((GROQ_MODEL, os.environ["GROQ_API_KEY"], GROQ_BASE_URL))
    if os.getenv("GEMINI_API_KEY"):
        backups.append((GEMINI_MODEL, os.environ["GEMINI_API_KEY"], GEMINI_BASE_URL))
    if not backups:
        return ChatOpenAI(model=OPENROUTER_MODEL, api_key=os.environ["OPENROUTER_API_KEY"],
                          base_url=OPENROUTER_BASE_URL, temperature=temperature)

    # Every model except the last has retries off, so a failure (e.g. the daily
    # quota) moves on to the next model at once.
    chain = [(OPENROUTER_MODEL, os.environ["OPENROUTER_API_KEY"], OPENROUTER_BASE_URL), *backups]
    models = [
        ChatOpenAI(model=name, api_key=key, base_url=base, temperature=temperature,
                   max_retries=0 if i < len(chain) - 1 else 2)
        for i, (name, key, base) in enumerate(chain)
    ]
    return _FallbackChatModel(models)
