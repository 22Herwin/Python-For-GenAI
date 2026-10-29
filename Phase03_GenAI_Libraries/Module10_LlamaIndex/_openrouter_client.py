# Module 10 - LlamaIndex
# Shared model factory - lets every 0X_*.py file in this folder configure
# LlamaIndex's Settings.llm / Settings.embed_model against a real OPENAI_API_KEY,
# or through OPENROUTER_API_KEY / a local OMNIROUTE_API_KEY gateway instead,
# with no other code changes needed in the calling file. If no key is set at all,
# an offline LLM + embedding model (deterministic, no network) can be used so the
# core logic of every file can still be exercised.
#
# NOTE: needs `pip install llama-index` (core + OpenAI LLM/embedding integrations).
# Routing through OpenRouter / OmniRoute additionally needs
# `pip install llama-index-llms-openai-like llama-index-embeddings-openai-like`,
# because the plain OpenAI classes reject non-OpenAI model names.
# OmniRoute's local gateway only proxies embeddings if an "openai" provider has
# been configured with real credentials in its own dashboard
# (http://127.0.0.1:20128); OPENROUTER_API_KEY is the confirmed-working option.
# This module imports llama_index lazily, so it can be imported without it.

import os
import re
import sys
import hashlib
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return. Force UTF-8 stdout so a print() of
# model output never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# An explicitly configured OmniRoute key takes precedence over other providers.
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
_USE_OMNIROUTE = bool(os.getenv("OMNIROUTE_API_KEY"))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_USE_OPENROUTER = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))

# Model strings. A direct OPENAI_API_KEY keeps the book's models unchanged.
if _USE_OMNIROUTE:
    LLM_MODEL = os.getenv("OMNIROUTE_MODEL", "cc/claude-sonnet-4-6")
    EMBEDDING_MODEL = os.getenv("OMNIROUTE_EMBEDDING_MODEL", "openai/text-embedding-3-small")
elif _USE_OPENROUTER:
    LLM_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
else:
    LLM_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"

# Context window reported to LlamaIndex for gateway models (OpenAILike cannot
# look it up by name the way the plain OpenAI class does).
GATEWAY_CONTEXT_WINDOW = 128_000


def has_llm_access() -> bool:
    """True when any of OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY is set."""
    return _USE_OMNIROUTE or bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def _credentials() -> tuple[str, str | None] | None:
    """(api_key, api_base) for the active provider; api_base is None for direct OpenAI."""
    if _USE_OMNIROUTE:
        return os.environ["OMNIROUTE_API_KEY"], OMNIROUTE_BASE_URL
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], None
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"], OPENROUTER_BASE_URL
    return None


def _missing_package(pip_name: str) -> None:
    raise SystemExit(f"Missing package. Install it with: pip install {pip_name}") from None


# ── Real models ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1, max_tokens: int = 1024):
    """LlamaIndex LLM for the active provider, or None when no key is set."""
    creds = _credentials()
    if creds is None:
        return None
    api_key, api_base = creds

    if api_base is None:
        try:
            from llama_index.llms.openai import OpenAI
        except ImportError:
            _missing_package("llama-index-llms-openai")
        return OpenAI(model=LLM_MODEL, api_key=api_key, temperature=temperature, max_tokens=max_tokens)

    try:
        from llama_index.llms.openai_like import OpenAILike
    except ImportError:
        _missing_package("llama-index-llms-openai-like")
    extra: dict[str, Any] = {"timeout": 120.0, "max_retries": 0} if _USE_OMNIROUTE else {}
    return OpenAILike(
        model=LLM_MODEL,
        api_key=api_key,
        api_base=api_base,
        is_chat_model=True,
        context_window=GATEWAY_CONTEXT_WINDOW,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )


def get_embed_model():
    """LlamaIndex embedding model for the active provider, or None when no key is set."""
    creds = _credentials()
    if creds is None:
        return None
    api_key, api_base = creds

    if api_base is None:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError:
            _missing_package("llama-index-embeddings-openai")
        return OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=api_key)

    try:
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    except ImportError:
        _missing_package("llama-index-embeddings-openai-like")
    return OpenAILikeEmbedding(model_name=EMBEDDING_MODEL, api_key=api_key, api_base=api_base)


# ── Offline models (no network) ──────────────────────────────────────────────

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the this to was what "
    "when where which who why with does do did can i you we they not".split()
)


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens with a crude plural strip and stopwords removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOPWORDS]


def _hash_vector(text: str, dim: int = 256) -> list[float]:
    """Deterministic bag-of-words hashing embedding, L2-normalised. Texts that share
    words get a higher cosine score - good enough to test retrieval logic offline,
    but NOT semantically meaningful like a real embedding model."""
    vector = [0.0] * dim
    for token in _tokens(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


_CONTEXT_BLOCK_RE = re.compile(r"-{5,}[ \t]*\n(.*?)\n[ \t]*-{5,}", re.S)
_QUERY_RE = re.compile(r"(?:Query|Question|original query is as follows):\s*(.*?)\s*(?:\n\s*(?:Answer|We have)|$)", re.S)
_METADATA_LINE_RE = re.compile(r"^[a-z_]+: [^\n]*$", re.M)


def _extractive_answer(prompt: str) -> str:
    """What the offline LLM 'answers': the 1-2 context sentences that share the
    most words with the query. Parses LlamaIndex's default QA/refine/summarise
    prompts (context between dashed lines, 'Query: ...' afterwards)."""
    blocks = _CONTEXT_BLOCK_RE.findall(prompt)
    queries = _QUERY_RE.findall(prompt)
    context = blocks[0] if blocks else prompt
    query = queries[-1] if queries else ""

    context = _METADATA_LINE_RE.sub("", context)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", context) if s.strip()]
    if not sentences:
        return "[offline mock LLM] (no context)"

    query_tokens = set(_tokens(query))
    scored = [(len(query_tokens & set(_tokens(s))), i, s) for i, s in enumerate(sentences)]
    best = sorted((t for t in scored if t[0] > 0), key=lambda t: (-t[0], t[1]))[:2]
    chosen = [s for _, _, s in sorted(best, key=lambda t: t[1])] or [sentences[0]]
    return "[offline mock LLM] " + " ".join(" ".join(chosen).split())[:400]


_OFFLINE_CACHE: dict[str, Any] = {}


def get_offline_llm():
    """Offline LlamaIndex LLM: answers by extracting the most relevant context
    sentences. Makes no network call."""
    if "llm" in _OFFLINE_CACHE:
        return _OFFLINE_CACHE["llm"]
    try:
        from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
        from llama_index.core.llms.callbacks import llm_completion_callback
    except ImportError:
        _missing_package("llama-index")

    class OfflineExtractiveLLM(CustomLLM):
        context_window: int = 3900
        num_output: int = 256
        model_name: str = "offline-extractive-llm"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=self.context_window,
                num_output=self.num_output,
                model_name=self.model_name,
            )

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            return CompletionResponse(text=_extractive_answer(prompt))

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
            text = _extractive_answer(prompt)
            yield CompletionResponse(text=text, delta=text)

    _OFFLINE_CACHE["llm"] = OfflineExtractiveLLM()
    return _OFFLINE_CACHE["llm"]


def get_offline_embed_model(dim: int = 256):
    """Offline LlamaIndex embedding model (hashed bag-of-words). Makes no network call."""
    key = f"embed_{dim}"
    if key in _OFFLINE_CACHE:
        return _OFFLINE_CACHE[key]
    try:
        from llama_index.core.embeddings import BaseEmbedding
    except ImportError:
        _missing_package("llama-index")

    class HashEmbedding(BaseEmbedding):
        dim: int = 256

        def _get_query_embedding(self, query: str) -> list[float]:
            return _hash_vector(query, self.dim)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return _hash_vector(query, self.dim)

        def _get_text_embedding(self, text: str) -> list[float]:
            return _hash_vector(text, self.dim)

    _OFFLINE_CACHE[key] = HashEmbedding(model_name="offline-hash-embedding", dim=dim)
    return _OFFLINE_CACHE[key]


# ── Settings helpers ─────────────────────────────────────────────────────────

def _apply_chunking(chunk_size: int | None, chunk_overlap: int | None) -> None:
    from llama_index.core import Settings

    if chunk_size is not None:
        Settings.chunk_size = chunk_size
    if chunk_overlap is not None:
        Settings.chunk_overlap = chunk_overlap


def configure_settings(
    temperature: float = 0.1,
    max_tokens: int = 1024,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> bool:
    """Point Settings.llm / Settings.embed_model at the real provider.
    Returns False (and changes nothing) when no API key is set."""
    if not has_llm_access():
        return False
    from llama_index.core import Settings

    Settings.llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    Settings.embed_model = get_embed_model()
    _apply_chunking(chunk_size, chunk_overlap)
    return True


def configure_offline_settings(chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
    """Point Settings at the offline LLM + hashing embedding model."""
    from llama_index.core import Settings

    Settings.llm = get_offline_llm()
    Settings.embed_model = get_offline_embed_model()
    _apply_chunking(chunk_size, chunk_overlap)


def configure_settings_or_offline(
    temperature: float = 0.1,
    max_tokens: int = 1024,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> str:
    """Configure Settings with the real provider if a key exists, otherwise the
    offline models. Returns "real" or "offline" and prints which one is active."""
    if configure_settings(temperature, max_tokens, chunk_size, chunk_overlap):
        print(f"[real] LLM={LLM_MODEL} | embeddings={EMBEDDING_MODEL}")
        return "real"
    configure_offline_settings(chunk_size, chunk_overlap)
    print(
        "[offline] No OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY set - "
        "using the offline mock LLM + hashing embeddings (answers are extractive, not generated). "
        "Add a key to a .env file in this folder for real model output."
    )
    return "offline"


def get_models_or_offline() -> tuple[Any, Any, str]:
    """(llm, embed_model, "real" | "offline") without touching global Settings."""
    if has_llm_access():
        return get_llm(), get_embed_model(), "real"
    return get_offline_llm(), get_offline_embed_model(), "offline"
