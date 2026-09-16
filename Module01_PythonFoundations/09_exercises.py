# Module 01 - Python Foundations
# 1.7 Module 01 Exercises
#
# All four exercises solved below. Run this file directly to see the results.

import functools
import time


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Write a function token_cost(tokens: int, model: str) -> float that returns
# the estimated cost using a dict of costs per 1K tokens inside the function.
# Raise a ValueError if the model is unknown.

def token_cost(tokens: int, model: str) -> float:
    cost_per_1k = {
        "gpt-4o":            2.50,
        "gpt-4o-mini":       0.15,
        "claude-sonnet-4-5": 3.00,
        "claude-haiku-4-5":  0.80,
    }
    if model not in cost_per_1k:
        raise ValueError(f"Unknown model: {model!r}")
    return round((tokens / 1000) * cost_per_1k[model], 6)


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Build a retry decorator (using functools.wraps) that retries a function up
# to n times on any exception, with a fixed sleep between attempts. Test it
# with a function that fails the first two times.

def retry(n: int = 3, delay: float = 0.05):
    """Parametrised retry decorator: @retry(n=3, delay=0.05)"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"  attempt {attempt} failed: {e}")
                    if attempt < n:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


_call_count = {"n": 0}


@retry(n=3, delay=0.01)
def flaky_function() -> str:
    """Fails on the first two calls, succeeds on the third."""
    _call_count["n"] += 1
    if _call_count["n"] < 3:
        raise ConnectionError(f"simulated failure #{_call_count['n']}")
    return "Success!"


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Write a temperature_label(t: float) -> str function that maps 0.0-0.3 to
# "precise", 0.3-0.7 to "balanced", 0.7-1.0 to "creative", and raises
# ValueError outside 0.0-1.0.

def temperature_label(t: float) -> str:
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"temperature must be between 0.0 and 1.0, got {t}")
    if t < 0.3:
        return "precise"
    elif t < 0.7:
        return "balanced"
    else:
        return "creative"


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Parse the string "128000 tokens, 0.005 USD per 1K" and extract the token
# count as an int and the cost as a float using only string methods (no regex).

def parse_token_cost_string(text: str) -> tuple[int, float]:
    tokens_part, cost_part = text.split(",")
    token_count = int(tokens_part.strip().split(" ")[0])
    cost = float(cost_part.strip().split(" ")[0])
    return token_count, cost


if __name__ == "__main__":
    print("=== Exercise 1: token_cost ===")
    print(token_cost(1500, "claude-sonnet-4-5"))   # 4.5
    try:
        token_cost(1000, "unknown-model")
    except ValueError as e:
        print(f"Raised as expected: {e}")

    print("\n=== Exercise 2: retry decorator ===")
    print(flaky_function())   # Success! (after 2 logged failures)

    print("\n=== Exercise 3: temperature_label ===")
    print(temperature_label(0.1))   # precise
    print(temperature_label(0.5))   # balanced
    print(temperature_label(0.9))   # creative
    try:
        temperature_label(1.5)
    except ValueError as e:
        print(f"Raised as expected: {e}")

    print("\n=== Exercise 4: parse_token_cost_string ===")
    tokens, cost = parse_token_cost_string("128000 tokens, 0.005 USD per 1K")
    print(f"tokens={tokens} ({type(tokens).__name__}), cost={cost} ({type(cost).__name__})")
