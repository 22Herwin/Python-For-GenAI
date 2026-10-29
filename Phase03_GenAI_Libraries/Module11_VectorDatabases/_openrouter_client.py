# Module 11 - Vector Databases
# Shared embedding helpers - lets every 0X_*.py file in this folder embed text
# against a real OPENAI_API_KEY, or through OPENROUTER_API_KEY / a local
# OMNIROUTE_API_KEY gateway instead, with no other code changes needed. When no
# key is set at all, get_embed_fn() falls back to a deterministic offline
# mock_embed() so the ChromaDB demos and exercises still run.
#
# NOTE: OmniRoute's local gateway only proxies embeddings if an "openai"
# provider has been configured with real credentials in its own dashboard
# (http://127.0.0.1:20128) - OMNIROUTE_API_KEY alone is not enough for this
# endpoint. If that's not set up, embed calls will fail with a clear
# "No credentials for embedding provider" error from OmniRoute itself; set
# OPENROUTER_API_KEY instead in that case.

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

OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
_USE_OMNIROUTE = bool(os.getenv("OMNIROUTE_API_KEY"))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_USE_OPENROUTER = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))

# Model string to pass to client.embeddings.create(model=...).
if _USE_OMNIROUTE:
    EMBEDDING_MODEL = os.getenv("OMNIROUTE_EMBEDDING_MODEL", "openai/text-embedding-3-small")
elif _USE_OPENROUTER:
    EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
else:
    EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_DIM = 1536   # text-embedding-3-small output dimension

EmbedFn = Callable[[list[str]], list[list[float]]]


def has_openai_access() -> bool:
    return _USE_OMNIROUTE or bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def get_openai_client():
    """Real OpenAI client, or the same SDK pointed at OmniRoute/OpenRouter -
    both are OpenAI-compatible, so no request/response translation is needed."""
    from openai import OpenAI

    if _USE_OMNIROUTE:
        return OpenAI(api_key=os.environ["OMNIROUTE_API_KEY"], base_url=OMNIROUTE_BASE_URL,
                      timeout=120.0, max_retries=0)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
    return None


def openai_embed(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """The embed() helper used throughout the book - one batched API call.
    Needs OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY."""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("No embedding API key set (OPENAI_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY).")
    resp = client.embeddings.create(model=model, input=texts)
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]


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
