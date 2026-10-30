# Module 10 - LlamaIndex
# Shared model factory - lets every 0X_*.py file in this folder configure
# LlamaIndex's Settings.llm / Settings.embed_model against a real OPENAI_API_KEY,
# or through a single OPENROUTER_API_KEY instead, with no other code changes
# needed in the calling file. If GEMINI_API_KEY and/or GROQ_API_KEY are also set,
# they act as backups behind OpenRouter: when it fails (e.g. its free daily quota
# is used up) the same call is retried on Gemini, then Groq (chat only - Groq has
# no embeddings). If no key is set at all, an offline LLM + embedding model
# (deterministic, no network) can be used so the core logic of every file can
# still be exercised.
#
# NOTE: needs `pip install llama-index` (core + OpenAI LLM/embedding integrations).
# Routing through OpenRouter / Gemini / Groq additionally needs
# `pip install llama-index-llms-openai-like llama-index-embeddings-openai-like`,
# because the plain OpenAI classes reject non-OpenAI model names.
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_USE_OPENROUTER = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))

# Backups behind OpenRouter, each used only when the one before it fails and only
# if its key is set. Groq has no embeddings endpoint, so it backs up chat only.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Backup models may be "reasoning" models that spend part of max_tokens on
# thinking, so they get at least this budget or the visible answer can come back empty.
BACKUP_MIN_TOKENS = 4096

# Model strings. A direct OPENAI_API_KEY keeps the book's models unchanged.
if _USE_OPENROUTER:
    LLM_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
else:
    LLM_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"

# Context window reported to LlamaIndex for gateway models (OpenAILike cannot
# look it up by name the way the plain OpenAI class does).
GATEWAY_CONTEXT_WINDOW = 128_000


def has_llm_access() -> bool:
    """True when OPENAI_API_KEY or OPENROUTER_API_KEY is set."""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def _credentials() -> tuple[str, str | None] | None:
    """(api_key, api_base) for the active provider; api_base is None for direct OpenAI."""
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

    def gateway_llm(model: str, key: str, base: str, tokens: int, retries: int):
        return OpenAILike(
            model=model,
            api_key=key,
            api_base=base,
            is_chat_model=True,
            context_window=GATEWAY_CONTEXT_WINDOW,
            temperature=temperature,
            max_tokens=tokens,
            max_retries=retries,
        )

    # (model, key, base, tokens) for each backup that has a key, in fallback order.
    specs = []
    if os.getenv("GEMINI_API_KEY"):
        specs.append((GEMINI_MODEL, os.environ["GEMINI_API_KEY"], GEMINI_BASE_URL,
                      max(max_tokens, BACKUP_MIN_TOKENS)))
    if os.getenv("GROQ_API_KEY"):
        specs.append((GROQ_MODEL, os.environ["GROQ_API_KEY"], GROQ_BASE_URL,
                      max(max_tokens, BACKUP_MIN_TOKENS)))
    if not specs:
        return gateway_llm(LLM_MODEL, api_key, api_base, max_tokens, retries=3)

    # Every model except the last has retries off, so a failure (e.g. the daily
    # quota) moves on to the next model at once instead of retrying and logging.
    models = [gateway_llm(LLM_MODEL, api_key, api_base, max_tokens, retries=0)]
    models += [gateway_llm(*spec, retries=0) for spec in specs[:-1]]
    models.append(gateway_llm(*specs[-1], retries=3))
    return _fallback_llm(models)


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
    if not os.getenv("GEMINI_API_KEY"):
        return OpenAILikeEmbedding(model_name=EMBEDDING_MODEL, api_key=api_key, api_base=api_base)
    # Retries off on OpenRouter so a failure switches to Gemini at once, silently.
    primary = OpenAILikeEmbedding(model_name=EMBEDDING_MODEL, api_key=api_key, api_base=api_base,
                                  max_retries=0)
    backup = OpenAILikeEmbedding(model_name=GEMINI_EMBEDDING_MODEL,
                                 api_key=os.environ["GEMINI_API_KEY"], api_base=GEMINI_BASE_URL)
    return _fallback_embed_model([primary, backup])


# ── Fallback wrappers ────────────────────────────────────────────────────────
# LlamaIndex has no with_fallbacks(), so these wrap several models behind one.

def _fallback_llm(llms: list[Any]):
    """One LLM that tries each of `llms` in order until one answers."""
    from llama_index.core.llms import (
        ChatResponseGen,
        CompletionResponseGen,
        CustomLLM,
        LLMMetadata,
    )
    from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

    state = {"llm": llms[0]}   # model that answered last

    def first_ok(call):
        """Run call(llm) on each model; re-raise the last error if all of them fail."""
        error: Exception | None = None
        for llm in llms:
            try:
                result = call(llm)
            except Exception as exc:
                error = exc
                continue
            state["llm"] = llm
            return result
        raise error

    def first_ok_stream(call):
        """Like first_ok for generators; only falls back before the first chunk."""
        error: Exception | None = None
        for llm in llms:
            started = False
            try:
                for chunk in call(llm):
                    if not started:
                        state["llm"] = llm
                    started = True
                    yield chunk
                return
            except Exception as exc:
                if started:
                    raise
                error = exc
        raise error

    class FallbackLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return state["llm"].metadata

        @property
        def active_model_name(self) -> str:
            return state["llm"].metadata.model_name

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            return first_ok(lambda llm: llm.complete(prompt, formatted=formatted, **kwargs))

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
            return first_ok_stream(lambda llm: llm.stream_complete(prompt, formatted=formatted, **kwargs))

        @llm_chat_callback()
        def chat(self, messages, **kwargs: Any):
            return first_ok(lambda llm: llm.chat(messages, **kwargs))

        @llm_chat_callback()
        def stream_chat(self, messages, **kwargs: Any) -> ChatResponseGen:
            return first_ok_stream(lambda llm: llm.stream_chat(messages, **kwargs))

    return FallbackLLM()


def _fallback_embed_model(models: list[Any]):
    """One embedding model that uses models[0] until it fails, then switches to the
    next for the rest of the run (vectors from two models must not be mixed)."""
    from llama_index.core.embeddings import BaseEmbedding
    from pydantic import PrivateAttr

    class FallbackEmbedding(BaseEmbedding):
        _active: int = PrivateAttr(default=0)

        @property
        def active_model_name(self) -> str:
            return models[self._active].model_name

        def _run(self, method: str, arg: Any) -> Any:
            while True:
                try:
                    return getattr(models[self._active], method)(arg)
                except Exception:
                    if self._active >= len(models) - 1:
                        raise
                    self._active += 1
                    self.model_name = models[self._active].model_name

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._run("get_query_embedding", query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._run("get_query_embedding", query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._run("get_text_embedding", text)

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self._run("get_text_embedding_batch", texts)

    return FallbackEmbedding(model_name=models[0].model_name)


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


def _active_model_names() -> tuple[str, str]:
    """Names of the models that actually answer. When backups are configured, one
    tiny call each finds out which provider is up (a failed probe only affects the
    printed label - the real calls report their own errors)."""
    from llama_index.core import Settings

    llm, embed = Settings.llm, Settings.embed_model
    if hasattr(llm, "active_model_name"):
        try:
            llm.complete("Reply with: OK")
        except Exception:
            pass
    if hasattr(embed, "active_model_name"):
        try:
            embed.get_text_embedding("ping")
        except Exception:
            pass
    return (getattr(llm, "active_model_name", LLM_MODEL),
            getattr(embed, "active_model_name", EMBEDDING_MODEL))


def configure_settings_or_offline(
    temperature: float = 0.1,
    max_tokens: int = 1024,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> str:
    """Configure Settings with the real provider if a key exists, otherwise the
    offline models. Returns "real" or "offline" and prints which one is active."""
    if configure_settings(temperature, max_tokens, chunk_size, chunk_overlap):
        llm_name, embed_name = _active_model_names()
        print(f"[real] LLM={llm_name} | embeddings={embed_name}")
        return "real"
    configure_offline_settings(chunk_size, chunk_overlap)
    print(
        "[offline] No OPENAI_API_KEY / OPENROUTER_API_KEY set - "
        "using the offline mock LLM + hashing embeddings (answers are extractive, not generated). "
        "Add a key to a .env file in this folder for real model output."
    )
    return "offline"


def get_models_or_offline() -> tuple[Any, Any, str]:
    """(llm, embed_model, "real" | "offline") without touching global Settings."""
    if has_llm_access():
        return get_llm(), get_embed_model(), "real"
    return get_offline_llm(), get_offline_embed_model(), "offline"
