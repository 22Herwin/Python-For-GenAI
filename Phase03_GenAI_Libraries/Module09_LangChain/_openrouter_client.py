# Module 09 - LangChain
# Shared model factory - lets every 0X_*.py file in this folder build a
# LangChain chat model (and embeddings) from a real OPENAI_API_KEY /
# ANTHROPIC_API_KEY, or from a single OPENROUTER_API_KEY, with no other code
# changes needed in the calling file. OMNIROUTE_API_KEY takes precedence and
# routes ChatOpenAI to the local gateway (OMNIROUTE_MODEL selects its model).
#
# OpenRouter and OmniRoute only expose an OpenAI-compatible endpoint, so when
# no ANTHROPIC_API_KEY is set, get_chat_model("anthropic") returns a ChatOpenAI
# pointed at the gateway with a Claude model id instead of ChatAnthropic.
#
# NOTE: Anthropic has no embeddings API, so get_embeddings() only works with an
# OpenAI-compatible key (OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY).
# OmniRoute only proxies embeddings if an "openai" provider is configured in its
# own dashboard; set OPENROUTER_API_KEY instead if that is not set up.

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return (emoji, arrows, checkmarks). Force
# UTF-8 stdout so a print(response_text) never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# An explicitly configured OmniRoute key takes precedence over other providers.
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
_USE_OMNIROUTE = bool(os.getenv("OMNIROUTE_API_KEY"))
OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "cc/claude-sonnet-4-6")
OMNIROUTE_EMBEDDING_MODEL = os.getenv("OMNIROUTE_EMBEDDING_MODEL", "openai/text-embedding-3-small")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_OPENAI_MODEL = os.getenv("OPENROUTER_OPENAI_MODEL", "openai/gpt-4o")
OPENROUTER_CLAUDE_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
# Backup for get_chat_model("auto") on OpenRouter: when the primary model errors
# (429, 5xx) or returns something the chain cannot parse (e.g. prose instead of
# JSON for with_structured_output), the same call is retried on this model.
# Set OPENROUTER_FALLBACK_MODEL= (empty) in .env to disable.
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3.5-super-vl:free")
# Last resort when every OpenRouter free model is exhausted (daily quota): Gemini
# through Google's OpenAI-compatible endpoint. Only used when GEMINI_API_KEY is set,
# and only as a backup behind OpenRouter (chat via "auto", and embeddings).
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Model ids used with a real provider key (same ones the guide uses).
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
EMBEDDING_MODEL = "text-embedding-3-small"

NO_LLM_KEY_MESSAGE = (
    "OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY (or ANTHROPIC_API_KEY) "
    "is not set. Add one to a .env file in this folder."
)
NO_EMBEDDINGS_KEY_MESSAGE = (
    "OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY is not set (embeddings need "
    "an OpenAI-compatible key). Add one to a .env file in this folder."
)

# Python module name -> pip package name, used for the "missing library" message.
_PIP_NAMES = {
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "langchain_anthropic": "langchain-anthropic",
    "langchain_community": "langchain-community",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_chroma": "langchain-chroma",
    "chromadb": "chromadb",
    "pydantic": "pydantic",
}
_BACKEND_MODULE = {"openai": "langchain_openai", "anthropic": "langchain_anthropic"}


# ── Provider routing ───────────────────────────────────────────────────────────

def _openai_route(want_claude: bool) -> tuple[str, str | None, str] | None:
    """(api_key, base_url, model) for an OpenAI-compatible chat endpoint, or None.
    With want_claude=True a direct OPENAI_API_KEY is skipped (it can't serve Claude)."""
    if _USE_OMNIROUTE:
        return os.environ["OMNIROUTE_API_KEY"], OMNIROUTE_BASE_URL, OMNIROUTE_MODEL
    if not want_claude and os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], None, OPENAI_MODEL
    if os.getenv("OPENROUTER_API_KEY"):
        model = OPENROUTER_CLAUDE_MODEL if want_claude else OPENROUTER_OPENAI_MODEL
        return os.environ["OPENROUTER_API_KEY"], OPENROUTER_BASE_URL, model
    return None


def _embeddings_route() -> tuple[str, str | None, str] | None:
    """(api_key, base_url, model) for an OpenAI-compatible embeddings endpoint, or None."""
    if _USE_OMNIROUTE:
        return os.environ["OMNIROUTE_API_KEY"], OMNIROUTE_BASE_URL, OMNIROUTE_EMBEDDING_MODEL
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], None, EMBEDDING_MODEL
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"], OPENROUTER_BASE_URL, OPENROUTER_EMBEDDING_MODEL
    return None


def chat_backend(provider: str = "auto") -> str | None:
    """Which LangChain integration get_chat_model(provider) would use:
    "openai" (ChatOpenAI), "anthropic" (ChatAnthropic) or None (no key set).
    provider is "openai", "anthropic" or "auto" (OpenAI-compatible first)."""
    if provider == "anthropic":
        if _USE_OMNIROUTE:
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "openai" if os.getenv("OPENROUTER_API_KEY") else None
    if provider == "openai":
        return "openai" if _openai_route(want_claude=False) else None
    if provider == "auto":
        if _openai_route(want_claude=False):
            return "openai"
        return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else None
    raise ValueError(f"Unknown provider {provider!r} (use 'openai', 'anthropic' or 'auto').")


def has_llm_access() -> bool:
    return chat_backend("auto") is not None


def has_embeddings_access() -> bool:
    return _embeddings_route() is not None


# ── Model factories ────────────────────────────────────────────────────────────

def get_chat_model(
    provider: str = "auto",
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Any | None:
    """Build a LangChain chat model, or return None when no key is configured.

    ChatOpenAI for direct OpenAI keys / OpenRouter / OmniRoute, ChatAnthropic for
    a direct ANTHROPIC_API_KEY. Raises ImportError if the needed integration
    package (langchain-openai / langchain-anthropic) is not installed."""
    backend = chat_backend(provider)
    if backend is None:
        return None

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or ANTHROPIC_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    from langchain_openai import ChatOpenAI

    route = _openai_route(want_claude=(provider == "anthropic"))
    assert route is not None   # chat_backend() already confirmed a route exists
    api_key, base_url, default_model = route
    extra: dict[str, Any] = {"timeout": 120.0, "max_retries": 0} if _USE_OMNIROUTE else {}
    chosen = model or default_model
    primary = ChatOpenAI(
        model=chosen,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        **extra,
    )

    # Only "auto" on OpenRouter gets backups; explicit "openai"/"anthropic"
    # callers keep a plain ChatOpenAI so the two models stay comparable.
    if provider == "auto" and base_url == OPENROUTER_BASE_URL:
        backups = []
        if OPENROUTER_FALLBACK_MODEL and chosen != OPENROUTER_FALLBACK_MODEL:
            backups.append(ChatOpenAI(
                model=OPENROUTER_FALLBACK_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
            ))
        if os.getenv("GEMINI_API_KEY"):
            backups.append(ChatOpenAI(
                model=GEMINI_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.environ["GEMINI_API_KEY"],
                base_url=GEMINI_BASE_URL,
            ))
        if backups:
            return primary.with_fallbacks(backups)
    return primary


def get_embeddings() -> Any | None:
    """OpenAIEmbeddings (direct, OpenRouter or OmniRoute), or None without a key."""
    route = _embeddings_route()
    if route is None:
        return None

    from langchain_openai import OpenAIEmbeddings

    api_key, base_url, model = route
    if base_url is None:
        return OpenAIEmbeddings(model=model, api_key=api_key)
    # Gateways expect plain text input; the default client-side tokenisation sends
    # token ids, which non-OpenAI backends reject.
    primary = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url,
                               check_embedding_ctx_length=False)
    if base_url != OPENROUTER_BASE_URL or not os.getenv("GEMINI_API_KEY"):
        return primary

    from langchain_core.embeddings import Embeddings

    backup = OpenAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, api_key=os.environ["GEMINI_API_KEY"],
                              base_url=GEMINI_BASE_URL, check_embedding_ctx_length=False)

    class FallbackEmbeddings(Embeddings):
        """OpenRouter first; after its first failure (e.g. daily free quota) switch to
        Gemini for the rest of the run, so vectors from two models are not mixed."""

        def __init__(self) -> None:
            self._use_backup = False

        def _call(self, method: str, arg: Any) -> Any:
            if not self._use_backup:
                try:
                    return getattr(primary, method)(arg)
                except Exception as exc:
                    self._use_backup = True
                    print(f"[embeddings] OpenRouter failed ({type(exc).__name__}); "
                          f"switching to Gemini {GEMINI_EMBEDDING_MODEL}", file=sys.stderr)
            return getattr(backup, method)(arg)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._call("embed_documents", texts)

        def embed_query(self, text: str) -> list[float]:
            return self._call("embed_query", text)

    return FallbackEmbeddings()


# ── Readiness checks (graceful failure instead of a traceback) ─────────────────

def _module_missing(spec: str) -> bool:
    """spec is a module name, or "a|b" meaning either module is enough."""
    return all(importlib.util.find_spec(name) is None for name in spec.split("|"))


def _problems(
    modules: Iterable[str],
    llm: str | tuple[str, ...] | None,
    embeddings: bool,
) -> list[str]:
    problems: list[str] = []
    required = list(modules)

    if llm is not None:
        providers = (llm,) if isinstance(llm, str) else tuple(llm)
        backends = {b for b in (chat_backend(p) for p in providers) if b is not None}
        if not backends:
            problems.append(NO_LLM_KEY_MESSAGE)
        required += [_BACKEND_MODULE[b] for b in sorted(backends)]

    if embeddings:
        if has_embeddings_access():
            required.append("langchain_openai")
        else:
            problems.append(NO_EMBEDDINGS_KEY_MESSAGE)

    for spec in dict.fromkeys(required):
        if _module_missing(spec):
            pip_names = " or ".join(_PIP_NAMES.get(name, name) for name in spec.split("|"))
            problems.append(f"Missing library: {pip_names}. Install it with: pip install {pip_names.split(' or ')[0]}")
    return problems


def ensure_ready(
    modules: Iterable[str] = (),
    llm: str | tuple[str, ...] | None = None,
    embeddings: bool = False,
) -> bool:
    """True when everything a script needs is available. Otherwise prints what is
    missing (libraries and/or API key) and returns False, so __main__ can exit
    cleanly instead of crashing with a traceback.

    modules    - importable module names ("a|b" = either one is enough)
    llm        - None, or provider(s) from chat_backend(): "auto", "openai", "anthropic"
    embeddings - True if get_embeddings() will be used"""
    problems = _problems(modules, llm, embeddings)
    for problem in problems:
        print(problem)
    return not problems


def is_ready(
    modules: Iterable[str] = (),
    llm: str | tuple[str, ...] | None = None,
    embeddings: bool = False,
) -> bool:
    """Same check as ensure_ready() but silent - for optional 'real LLM' sections."""
    return not _problems(modules, llm, embeddings)


# ── Offline fake model ─────────────────────────────────────────────────────────

def make_fake_llm(respond: Callable[[str], str]):
    """A stand-in chat model for offline demos and tests. It is a Runnable: it
    takes whatever a prompt produces, passes the rendered text to respond(), and
    returns an AIMessage - so `prompt | fake_llm | StrOutputParser()` works with
    no network call and no API key."""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    def _call(prompt_value: Any) -> AIMessage:
        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        return AIMessage(content=respond(text))

    return RunnableLambda(_call)
