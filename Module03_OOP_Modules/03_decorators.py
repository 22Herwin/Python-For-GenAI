# Module 03 - OOP & Modules
# 3.3 Decorators
# Decorators wrap a function to add behaviour - logging, caching, retry, rate
# limiting - without modifying the function's body. They are used extensively
# in FastAPI, LangChain, and Python testing.

import functools
import time


def log_call(func):
    """Decorator: log function name and return value."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f">>> Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"<<< {func.__name__} returned: {result!r}")
        return result
    return wrapper


def cache_result(func):
    """Simple in-memory cache (no expiry)."""
    _cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in _cache:
            _cache[args] = func(*args)
        return _cache[args]
    return wrapper


@log_call
@cache_result
def get_embedding(text: str) -> list[float]:
    """Simulate an embedding API call (cached)."""
    time.sleep(0.01)   # simulate latency
    return [hash(text) % 100 / 100.0, 0.42, 0.87]


# First call: logs + computes
e1 = get_embedding("What is RAG?")
# Second call: logs but returns from cache instantly
e2 = get_embedding("What is RAG?")
print(e1 == e2)   # True


# ── Parametrised Decorators ────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 0.1):
    """Parametrised retry decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


@retry(max_attempts=3, delay=0.05)
def flaky_api_call(prompt: str) -> str:
    import random
    if random.random() < 0.6:   # fails 60% of the time
        raise ConnectionError("Simulated network error")
    return f"Response to: {prompt}"


if __name__ == "__main__":
    # flaky_api_call fails ~60% of the time per attempt, so even with 3
    # retries there is a small (~21.6%) chance all 3 attempts fail. That is
    # expected, realistic retry-decorator behaviour - not a bug - so the demo
    # just reports it instead of letting the traceback look alarming.
    try:
        print(flaky_api_call("What is a neural network?"))
    except ConnectionError as e:
        print(f"All retries exhausted (this can happen - {flaky_api_call.__name__} "
              f"fails 60% of the time per attempt): {e}")
