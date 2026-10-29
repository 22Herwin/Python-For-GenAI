# Module 13 - LangGraph & Agents
# Shared chat-model factory - lets every 0X_*.py file in this folder build a
# LangChain chat model from a real OPENAI_API_KEY, or through OPENROUTER_API_KEY
# / a local OMNIROUTE_API_KEY gateway instead, with no other code changes.
# Both gateways speak the OpenAI-compatible chat API, so ChatOpenAI is enough
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

# An explicitly configured OmniRoute key takes precedence over other providers.
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
_USE_OMNIROUTE = bool(os.getenv("OMNIROUTE_API_KEY"))
OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "cc/claude-sonnet-4-6")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_USE_OPENROUTER = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

# Model used when OPENAI_API_KEY is set (the model the book uses).
DEFAULT_MODEL = "gpt-4o"

LANGGRAPH_INSTALL_HINT = "pip install langgraph langchain-core langchain-openai python-dotenv"
NO_KEY_HINT = "OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY is not set. Add one to a .env file in this folder."


def has_llm_access() -> bool:
    """True if any supported API key / gateway key is configured."""
    return _USE_OMNIROUTE or bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    """LangChain ChatOpenAI routed through OmniRoute, direct OpenAI, or
    OpenRouter. Returns None when no key is configured.

    Raises ImportError (with an install hint) if langchain-openai is missing."""
    if not has_llm_access():
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(f"langchain-openai is not installed. Run: {LANGGRAPH_INSTALL_HINT}") from exc

    if _USE_OMNIROUTE:
        return ChatOpenAI(model=OMNIROUTE_MODEL, api_key=os.environ["OMNIROUTE_API_KEY"],
                          base_url=OMNIROUTE_BASE_URL, temperature=temperature,
                          timeout=120.0, max_retries=0)
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=model or DEFAULT_MODEL, api_key=os.environ["OPENAI_API_KEY"],
                          temperature=temperature)
    return ChatOpenAI(model=OPENROUTER_MODEL, api_key=os.environ["OPENROUTER_API_KEY"],
                      base_url=OPENROUTER_BASE_URL, temperature=temperature)
