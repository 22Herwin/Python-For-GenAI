# Module 14 - CrewAI
# Shared LLM factory - lets every 0X_*.py file in this folder run against a real
# OPENAI_API_KEY (the book's setup), or through OPENROUTER_API_KEY, or
# ANTHROPIC_API_KEY, with no other code changes needed in the calling file. If
# GEMINI_API_KEY and/or GROQ_API_KEY are also set (and OpenRouter is the provider),
# they back it up: when OpenRouter fails (e.g. its free daily quota is used up) the
# same call is retried on Gemini, then Groq.
#
# Precedence: OPENAI_API_KEY > OPENROUTER_API_KEY > ANTHROPIC_API_KEY.
#
# get_llm() builds the `llm` object that is passed to Agent(llm=...) as a
# crewai.LLM("provider/model", ...). The book passes a langchain ChatOpenAI, but
# current CrewAI releases only accept a model string or a crewai LLM there, so the
# OpenAI-compatible providers (OpenRouter, Gemini, Groq) are reached through
# crewai.LLM("openai/<model>", base_url=...).
#
# NOTE: CrewAI makes MANY LLM calls per kickoff (several per task - reasoning
# steps, tool calls, delegation), so every crew.kickoff() in this module makes
# REAL, billed API calls. Keep an eye on usage, especially for the hierarchical
# process in 06_hierarchical_process.py and 07_exercises.py.

import importlib
import importlib.util
import logging
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

class _DropTokenCountWarning(logging.Filter):
    """CrewAI warns on every call that it can only estimate input tokens for models
    it has no tokenizer for (anything that is not an OpenAI model). Harmless noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "approximate input token count" not in record.getMessage()


logging.getLogger("crewai.llms.base_llm").addFilter(_DropTokenCountWarning())

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

NO_KEY_MESSAGE = (
    "OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY "
    "is not set. Add one to a .env file in this folder (see .env.example)."
)

_DEFAULT_MODELS = {
    "openai": ("OPENAI_MODEL", "gpt-4o"),
    "openrouter": ("OPENROUTER_MODEL", "openai/gpt-4o"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
}

# Backups behind OpenRouter, each used only if its key is set, in this order. Groq goes
# before Gemini on purpose: agents replay earlier tool calls in the next request, and
# Gemini 3.x rejects that over the OpenAI-compatible API (it wants a "thought_signature"
# the client never sends), so Gemini can only back up single-turn calls.
_BACKUPS = [
    ("groq", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b", GROQ_BASE_URL),
    ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.1-flash-lite", GEMINI_BASE_URL),
]

# CrewAI asks "Share this execution trace with CrewAI? [y/N]" (20 s timeout) after
# every kickoff and can upload traces. Off by default here; set it to true to opt in.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

_LLM_CACHE: dict[str, Any] = {}


def _provider() -> str | None:
    """Which backend the current environment selects, or None without any key."""
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
    """Human-readable 'provider / model' label for printing in demos. With backups
    configured, one tiny call finds out which provider is actually answering."""
    provider = _provider()
    if provider is None:
        return "no LLM configured"
    llm = get_llm()
    if hasattr(llm, "active_label"):
        try:
            llm.call("Reply with: OK")
        except Exception:
            pass   # display only - the real calls report their own errors
        return llm.active_label
    return f"{provider} / {_model_name(provider)}"


def require_crewai() -> None:
    """Exit cleanly (no traceback) when the crewai package is not installed."""
    if importlib.util.find_spec("crewai") is None:
        raise SystemExit(
            "The 'crewai' package is not installed.\n"
            "Install it with:  pip install crewai python-dotenv"
        )


def load_sibling(module_name: str) -> ModuleType:
    """Import another numbered file of this module (e.g. '03_tools_for_agents').
    File names starting with a digit can't be used in a normal `import`
    statement, so importlib is used instead."""
    return importlib.import_module(module_name)


def _gateway_llm(model: str, key: str, base_url: str, retries: int | None = None) -> Any:
    """crewai.LLM for an OpenAI-compatible endpoint (OpenRouter, Gemini, Groq)."""
    from crewai import LLM

    extra = {} if retries is None else {"max_retries": retries}
    return LLM(model=f"openai/{model}", api_key=key, base_url=base_url, **extra)


def _native_llm(provider: str) -> Any:
    """crewai.LLM routed through its native providers ('provider/model')."""
    from crewai import LLM

    model = _model_name(provider)
    if provider == "openai":
        return LLM(model=f"openai/{model}", api_key=os.environ["OPENAI_API_KEY"])
    if provider == "openrouter":
        return _openrouter_with_backups(model)
    return LLM(model=f"anthropic/{model}", api_key=os.environ["ANTHROPIC_API_KEY"])


def _openrouter_with_backups(model: str) -> Any:
    chain = [("openrouter", model, os.environ["OPENROUTER_API_KEY"], OPENROUTER_BASE_URL)]
    for label, key_env, model_env, default, base in _BACKUPS:
        if os.getenv(key_env):
            chain.append((label, os.getenv(model_env, default), os.environ[key_env], base))
    if len(chain) == 1:
        label, name, key, base = chain[0]
        return _gateway_llm(name, key, base)

    # Every model except the last has SDK retries off, so a failure moves on at once.
    # The "LLM call failed" console panel is muted on all of them: if every model fails,
    # the wrapper raises the most useful error itself.
    entries = []
    for i, (label, name, key, base) in enumerate(chain):
        last = i == len(chain) - 1
        llm = _gateway_llm(name, key, base, retries=None if last else 0)
        object.__setattr__(llm, "_emit_call_failed_event", lambda *args, **kwargs: None)
        entries.append((label, name, llm))
    return _fallback_llm(entries)


def _fallback_llm(entries: list[tuple[str, str, Any]]) -> Any:
    """One crewai LLM that tries each (label, model, llm) in order until one answers."""
    from crewai.llms.base_llm import BaseLLM, call_stop_override

    state = {"index": 0}   # entry that answered last

    def active() -> Any:
        return entries[state["index"]][2]

    def is_throttle(exc: Exception) -> bool:
        return getattr(exc, "status_code", None) == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()

    def run_with_fallback(wrapper, call):
        errors: list[Exception] = []
        for index, (_, _, llm) in enumerate(entries):
            llm.stop = list(wrapper.stop)
            # CrewAI logs every failed attempt as an ERROR line; keep that quiet, since the
            # next model is tried at once (and the final error is raised below).
            previous = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                # Per-call stop words are keyed by LLM instance, so re-apply them to the inner one.
                with call_stop_override(llm, wrapper.stop_sequences):
                    result = call(llm)
            except Exception as exc:
                errors.append(exc)
                continue
            finally:
                logging.disable(previous)
            state["index"] = index
            return result
        # Every model failed. Prefer a rate-limit error: CrewAI retries those with a delay
        # (honouring Retry-After), while e.g. Gemini's tool-call 400 would just abort.
        throttles = [exc for exc in errors if is_throttle(exc)]
        raise (throttles or errors)[-1]

    class FallbackLLM(BaseLLM):
        llm_type: str = "fallback"

        @property
        def active_label(self) -> str:
            label, name, _ = entries[state["index"]]
            return f"{label} / {name}"

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            return run_with_fallback(self, lambda llm: llm.call(
                messages, tools=tools, callbacks=callbacks, available_functions=available_functions,
                from_task=from_task, from_agent=from_agent, response_model=response_model))

        def get_token_usage_summary(self):
            """Crew metrics read this: add up what every inner model has used."""
            from crewai.types.usage_metrics import UsageMetrics

            total = UsageMetrics()
            for _, _, llm in entries:
                total.add_usage_metrics(llm.get_token_usage_summary())
            return total

        def supports_function_calling(self) -> bool:
            return active().supports_function_calling()

        def supports_stop_words(self) -> bool:
            return active().supports_stop_words()

        def get_context_window_size(self) -> int:
            return active().get_context_window_size()

    return FallbackLLM(model=entries[0][1])


def get_llm(offline_ok: bool = False) -> Any | None:
    """LLM object for Agent(llm=...).

    Returns None when no API key is configured. With offline_ok=True it returns
    a placeholder crewai.LLM instead (never used for a real call) so Agent / Task /
    Crew objects can still be constructed and inspected without a key."""
    provider = _provider()
    if provider is None:
        return _native_llm_placeholder() if offline_ok else None
    if provider not in _LLM_CACHE:
        _LLM_CACHE[provider] = _native_llm(provider)
    return _LLM_CACHE[provider]


def _native_llm_placeholder() -> Any:
    from crewai import LLM

    return LLM(model="openai/gpt-4o", api_key="sk-offline-placeholder")
