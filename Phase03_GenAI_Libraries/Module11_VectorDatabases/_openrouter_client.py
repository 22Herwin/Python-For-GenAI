# Module 11 - Vector Databases
# Shared embedding helpers - lets every 0X_*.py file in this folder embed text
# against a real OPENAI_API_KEY, or through OPENROUTER_API_KEY instead, with no
# other code changes needed. If GEMINI_API_KEY is also set it backs up
# OpenRouter: when OpenRouter fails (e.g. its free daily quota is used up) the
# call is retried on Gemini and stays there for the rest of the run. When no key
# is set at all, get_embed_fn() falls back to a deterministic offline
# mock_embed() so the ChromaDB demos and exercises still run.
#
# NOTE: Gemini vectors have 3072 dimensions, not the 1536 of text-embedding-3-small
# (EMBEDDING_DIM below), so the Pinecone index in 03_pinecone.py only matches when
# the OpenAI model is the one answering.

import hashlib
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_USE_OPENROUTER = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))

# Backup behind OpenRouter, used only if GEMINI_API_KEY is set.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Model string to pass to client.embeddings.create(model=...).
if _USE_OPENROUTER:
    EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
else:
    EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_DIM = 1536   # text-embedding-3-small output dimension

EmbedFn = Callable[[list[str]], list[list[float]]]


def has_openai_access() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def _has_gemini_backup() -> bool:
    return _USE_OPENROUTER and bool(os.getenv("GEMINI_API_KEY"))


def get_openai_client():
    """Real OpenAI client, or the same SDK pointed at OpenRouter - both are
    OpenAI-compatible, so no request/response translation is needed."""
    from openai import OpenAI

    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if os.getenv("OPENROUTER_API_KEY"):
        # With a Gemini backup, a failure should switch at once instead of retrying.
        return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL,
                      max_retries=0 if _has_gemini_backup() else 2)
    return None


_use_gemini_backup = False   # set after OpenRouter fails; stays on for the rest of the run


def _vectors(resp) -> list[list[float]]:
    """Embeddings in input order. Sorted by `index` when the provider sets it
    (Gemini's OpenAI-compatible endpoint can leave it empty - then the order is already right)."""
    items = list(resp.data)
    if all(isinstance(e.index, int) for e in items):
        items.sort(key=lambda e: e.index)
    return [e.embedding for e in items]


def openai_embed(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """The embed() helper used throughout the book - one batched API call.
    Needs OPENAI_API_KEY / OPENROUTER_API_KEY; falls back to Gemini if configured."""
    global _use_gemini_backup
    client = get_openai_client()
    if client is None:
        raise RuntimeError("No embedding API key set (OPENAI_API_KEY / OPENROUTER_API_KEY).")

    if not _use_gemini_backup:
        try:
            return _vectors(client.embeddings.create(model=model, input=texts))
        except Exception:
            if not _has_gemini_backup():
                raise
            _use_gemini_backup = True   # vectors from two models must not be mixed in one run

    from openai import OpenAI

    gemini = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)
    return _vectors(gemini.embeddings.create(model=GEMINI_EMBEDDING_MODEL, input=texts))


def mock_embed(texts: list[str], dim: int = 64) -> list[list[float]]:
    """Deterministic offline embedding: signed hashed bag-of-words, L2-normalised.
    The same text always maps to the same vector (stable across processes, so it
    is safe to persist), and texts sharing words land close together - enough to
    exercise vector DB logic without any network call. Not a real semantic model."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float64)
        for token in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
            vec[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0], norm = 1.0, 1.0   # avoid zero vectors (undefined cosine distance)
        vectors.append((vec / norm).tolist())
    return vectors


def get_embed_fn() -> tuple[EmbedFn, str]:
    """Return (embed_fn, label): the real embedder when a key is available,
    otherwise the offline mock. The label is "openai" or "mock"."""
    if has_openai_access():
        return openai_embed, "openai"
    return mock_embed, "mock"
