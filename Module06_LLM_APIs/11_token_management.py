# Module 06 - OpenAI & Anthropic APIs
# 6.6 Token Management
# Tokens are the billing unit. Understanding token counts prevents surprises
# and helps you design prompts that stay within context limits.
#
# The count_tokens() call below needs a real ANTHROPIC_API_KEY (it's a free
# API call - no generation happens). estimate_cost() and fits_in_context()
# are pure Python and need no network access at all.

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


def count_tokens_demo() -> None:
    """Count tokens before sending (no generation happens, this call is free)."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        system="You are a concise assistant.",
        messages=[{"role": "user", "content": "Explain the transformer architecture."}],
    )
    print(f"Estimated input tokens: {response.input_tokens}")


# ── Cost estimator (pure Python - no API needed) ──────────────────────────────

PRICING = {
    "claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},  # per 1M tokens
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
    "gpt-4o":             {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":        {"input": 0.15,  "output": 0.60},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# ── Context window limits (always check before sending long documents) ───────

CONTEXT_LIMITS = {
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5":   200_000,
    "gpt-4o":             128_000,
    "gpt-4o-mini":        128_000,
    "gemini-1.5-pro":    1_000_000,
}


def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit


if __name__ == "__main__":
    if os.getenv("ANTHROPIC_API_KEY"):
        count_tokens_demo()
    else:
        print("ANTHROPIC_API_KEY is not set - skipping the live count_tokens() call.")

    print()
    cost = estimate_cost("claude-sonnet-4-5", input_tokens=500, output_tokens=300)
    print(f"Estimated cost: ${cost:.6f}")

    print(fits_in_context("gpt-4o", token_count=120_000))    # True
    print(fits_in_context("gpt-4o", token_count=127_000))    # False (would exceed with reserve)
