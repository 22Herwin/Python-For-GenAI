# Module 12 - Hugging Face Transformers
# 12.3 Tokenizers
# Understanding tokenizers is essential for managing context windows and
# debugging model behaviour. This file shows encoding/decoding, inspecting the
# individual tokens, batch encoding with padding + truncation, and counting
# tokens for a prompt before sending it to any model.
#
# NOTE: Needs `pip install transformers torch` (torch is only used for the
# return_tensors="pt" batch demo). The first run downloads the small GPT-2
# tokenizer files (~1 MB) from the Hugging Face Hub, so internet is required
# once; no model weights are downloaded. No GPU needed.
# NOTE: GPT-2 has no padding token, so batch encoding would raise an error as
# written in the book. We set pad_token = eos_token (the standard workaround).

import sys
from functools import lru_cache
from importlib.util import find_spec

from dotenv import load_dotenv

load_dotenv()

REQUIRED_PACKAGES = ["transformers", "torch"]


def missing_packages(packages: list[str]) -> list[str]:
    """Return the subset of `packages` that is not installed."""
    return [p for p in packages if find_spec(p) is None]


@lru_cache(maxsize=8)
def load_tokenizer(model: str = "gpt2"):
    """Load (and cache) the tokenizer for any model on the Hub."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 family has no pad token
    return tokenizer


# ── Encode / decode ───────────────────────────────────────────────────────────

def demo_encode_decode(model: str = "gpt2") -> None:
    tokenizer = load_tokenizer(model)

    text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."

    # Encode to token IDs
    tokens = tokenizer.encode(text)
    print(f"Token count: {len(tokens)}")
    print(f"Token IDs: {tokens}")

    # Decode back to text
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")

    # Tokenize to see individual tokens
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Tokens: {token_strings}")


# ── Batch encoding ────────────────────────────────────────────────────────────

def demo_batch_encoding(model: str = "gpt2") -> None:
    """Batch encoding with padding and truncation (for model input)."""
    tokenizer = load_tokenizer(model)

    batch = ["What is RAG?", "Explain vector databases in detail."]
    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",  # PyTorch tensors; use "tf" for TensorFlow
    )
    print(f"\nBatch input_ids shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")


# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt2") -> int:
    """Count tokens for a prompt before sending it to any model."""
    return len(load_tokenizer(model).encode(text))


def demo_count_tokens() -> None:
    prompt = "Summarise the following document: " + "word " * 500
    print(f"\nPrompt token count: {count_tokens(prompt)}")


if __name__ == "__main__":
    missing = missing_packages(REQUIRED_PACKAGES)
    if missing:
        print(f"Missing package(s): {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
        sys.exit(0)

    try:
        demo_encode_decode()
        demo_batch_encoding()
        demo_count_tokens()
    except OSError as exc:  # tokenizer files could not be downloaded / found
        print(f"Could not load the 'gpt2' tokenizer: {exc}")
        print("Check your internet connection (needed once to download it).")
