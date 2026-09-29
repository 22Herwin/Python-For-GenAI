# Module 08 - Embeddings & Semantic Search
# Shared client factory - lets every 0X_*.py file in this folder embed text
# against a real OPENAI_API_KEY, or through OPENROUTER_API_KEY / a local
# OMNIROUTE_API_KEY gateway instead, with no other code changes needed.
#
# NOTE: OmniRoute's local gateway only proxies embeddings if an "openai"
# provider has been configured with real credentials in its own dashboard
# (http://127.0.0.1:20128) - OMNIROUTE_API_KEY alone is not enough for this
# endpoint. If that's not set up, embed calls will fail with a clear
# "No credentials for embedding provider" error from OmniRoute itself; set
# OPENROUTER_API_KEY instead in that case (confirmed working).

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

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
