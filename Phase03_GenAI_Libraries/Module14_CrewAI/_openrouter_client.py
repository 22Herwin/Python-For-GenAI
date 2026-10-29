# Module 14 - CrewAI
# Shared LLM factory - lets every 0X_*.py file in this folder run against a real
# OPENAI_API_KEY (the book's setup), or through OPENROUTER_API_KEY, a local
# OMNIROUTE_API_KEY gateway, or ANTHROPIC_API_KEY, with no other code changes
# needed in the calling file.
#
# Precedence: OMNIROUTE_API_KEY > OPENAI_API_KEY > OPENROUTER_API_KEY > ANTHROPIC_API_KEY.
#
# get_llm() builds the `llm` object that is passed to Agent(llm=...):
# - Default: langchain_openai.ChatOpenAI, exactly like the book's snippets
#   (pointed at OpenRouter/OmniRoute with base_url when one of those keys is set).
# - CREWAI_LLM_STYLE=native (or ANTHROPIC_API_KEY only, or langchain-openai not
#   installed): crewai.LLM("provider/model", ...) instead, which CrewAI routes
#   through LiteLLM. Use this if your CrewAI version does not accept
#   ChatOpenAI objects for Agent(llm=...).
#
# NOTE: CrewAI makes MANY LLM calls per kickoff (several per task - reasoning
# steps, tool calls, delegation), so every crew.kickoff() in this module makes
# REAL, billed API calls. Keep an eye on usage, especially for the hierarchical
# process in 06_hierarchical_process.py and 07_exercises.py.

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return (emoji, arrows, checkmarks). Force
# UTF-8 stdout so a print(result) never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

NO_KEY_MESSAGE = (
    "OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY / ANTHROPIC_API_KEY "
    "is not set. Add one to a .env file in this folder (see .env.example)."
)

_DEFAULT_MODELS = {
    "omniroute": ("OMNIROUTE_MODEL", "cc/claude-sonnet-4-6"),
    "openai": ("OPENAI_MODEL", "gpt-4o"),
    "openrouter": ("OPENROUTER_MODEL", "openai/gpt-4o"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
}


def _provider() -> str | None:
    """Which backend the current environment selects, or None without any key."""
    if os.getenv("OMNIROUTE_API_KEY"):
        return "omniroute"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _model_name(provider: str) -> str:
    env_name, default = _DEFAULT_MODELS[provider]
    return os.getenv(env_name, default)


def has_llm_access() -> bool:
    return _provider() is not None


def describe_llm() -> str:
    """Human-readable 'provider / model' label for printing in demos."""
    provider = _provider()
    if provider is None:
        return "no LLM configured"
    return f"{provider} / {_model_name(provider)}"


def require_crewai() -> None:
    """Exit cleanly (no traceback) when the crewai package is not installed."""
    if importlib.util.find_spec("crewai") is None:
        raise SystemExit(
            "The 'crewai' package is not installed.\n"
            "Install it with:  pip install crewai langchain-openai python-dotenv"
        )


def load_sibling(module_name: str) -> ModuleType:
    """Import another numbered file of this module (e.g. '03_tools_for_agents').
    File names starting with a digit can't be used in a normal `import`
    statement, so importlib is used instead."""
    return importlib.import_module(module_name)


def _native_llm(provider: str) -> Any:
    """crewai.LLM routed through LiteLLM model strings ('provider/model')."""
    from crewai import LLM

    model = _model_name(provider)
    if provider == "omniroute":
        return LLM(model=f"openai/{model}", api_key=os.environ["OMNIROUTE_API_KEY"],
                   base_url=OMNIROUTE_BASE_URL, timeout=120.0)
    if provider == "openai":
        return LLM(model=f"openai/{model}", api_key=os.environ["OPENAI_API_KEY"])
    if provider == "openrouter":
        return LLM(model=f"openrouter/{model}", api_key=os.environ["OPENROUTER_API_KEY"],
                   base_url=OPENROUTER_BASE_URL)
    return LLM(model=f"anthropic/{model}", api_key=os.environ["ANTHROPIC_API_KEY"])


def _langchain_llm(provider: str) -> Any:
    """The book's `ChatOpenAI(model=..., api_key=...)`, optionally with a base_url
    for the OpenAI-compatible gateways."""
    from langchain_openai import ChatOpenAI

    model = _model_name(provider)
    if provider == "omniroute":
        return ChatOpenAI(model=model, api_key=os.environ["OMNIROUTE_API_KEY"],
                          base_url=OMNIROUTE_BASE_URL, timeout=120.0, max_retries=0)
    if provider == "openrouter":
        return ChatOpenAI(model=model, api_key=os.environ["OPENROUTER_API_KEY"],
                          base_url=OPENROUTER_BASE_URL)
    return ChatOpenAI(model=model, api_key=os.environ["OPENAI_API_KEY"])


def get_llm(offline_ok: bool = False) -> Any | None:
    """LLM object for Agent(llm=...).

    Returns None when no API key is configured. With offline_ok=True it returns
    a placeholder crewai.LLM instead (never used for a real call) so Agent / Task /
    Crew objects can still be constructed and inspected without a key."""
    provider = _provider()
    if provider is None:
        return _native_llm_placeholder() if offline_ok else None

    wants_native = os.getenv("CREWAI_LLM_STYLE", "").lower() == "native"
    if provider == "anthropic" or wants_native:
        return _native_llm(provider)
    if importlib.util.find_spec("langchain_openai") is None:
        print("[client] langchain-openai is not installed - using crewai.LLM instead of ChatOpenAI.")
        return _native_llm(provider)
    return _langchain_llm(provider)


def _native_llm_placeholder() -> Any:
    from crewai import LLM

    return LLM(model="openai/gpt-4o", api_key="sk-offline-placeholder")
