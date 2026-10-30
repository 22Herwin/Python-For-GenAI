# Module 03 - OOP & Modules
# 3.6 Module 03 Exercises
#
# Exercises 1-3 solved below. Exercise 4 (packaging ConversationHistory and
# LLMConfig into a real package with __init__.py exports) lives in the
# `exercise4_package/` folder next to this file - see exercise4_demo.py to
# run it.

import functools
import time
from dataclasses import dataclass, field
from string import Formatter


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Implement a RateLimiter class with a method check_and_wait() that ensures no
# more than N calls per minute, sleeping as needed. Test it by calling it 5
# times rapidly.

class RateLimiter:
    """Ensures no more than `max_calls_per_minute` calls happen in any rolling
    window. Sleeps inside check_and_wait() when the limit is hit.

    `window_seconds` defaults to 60.0 (a true "per minute" limiter) but is
    configurable so it can be demoed/tested without a real 60-second wait."""

    def __init__(self, max_calls_per_minute: int, window_seconds: float = 60.0):
        self.max_calls_per_minute = max_calls_per_minute
        self.window_seconds = window_seconds
        self._call_times: list[float] = []

    def check_and_wait(self) -> None:
        now = time.time()
        window_start = now - self.window_seconds
        self._call_times = [t for t in self._call_times if t > window_start]

        if len(self._call_times) >= self.max_calls_per_minute:
            oldest = self._call_times[0]
            wait_time = self.window_seconds - (now - oldest)
            if wait_time > 0:
                print(f"  Rate limit reached, sleeping {wait_time:.2f}s...")
                time.sleep(wait_time)
            now = time.time()
            window_start = now - self.window_seconds
            self._call_times = [t for t in self._call_times if t > window_start]

        self._call_times.append(time.time())


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Create a PromptTemplate dataclass with a template string field and a
# render(**kwargs) method that fills in placeholders using str.format_map.
# Add validation that all placeholders are provided.

@dataclass
class PromptTemplate:
    template: str
    required_vars: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        formatter = Formatter()
        self.required_vars = [
            fname for _, fname, _, _ in formatter.parse(self.template)
            if fname is not None
        ]

    def render(self, **kwargs) -> str:
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.template.format_map(kwargs)


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Write a @retry(max_attempts=3, delay=0.1) parametrised decorator. Test it on
# a function that raises on the first two calls.

def retry(max_attempts: int = 3, delay: float = 0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"  attempt {attempt} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


_call_counter = {"n": 0}


@retry(max_attempts=3, delay=0.05)
def fails_twice_then_succeeds() -> str:
    """Raises on the first two calls, succeeds on the third."""
    _call_counter["n"] += 1
    if _call_counter["n"] < 3:
        raise RuntimeError(f"simulated failure #{_call_counter['n']}")
    return "succeeded on 3rd try"


if __name__ == "__main__":
    print("=== Exercise 1: RateLimiter ===")
    # Demo uses a 2-second window (instead of the real 60s) so the enforced
    # wait below is short but still genuinely observable.
    limiter = RateLimiter(max_calls_per_minute=3, window_seconds=2.0)
    for i in range(1, 6):
        start = time.time()
        limiter.check_and_wait()
        elapsed = time.time() - start
        print(f"Call {i}: proceeded (waited {elapsed:.3f}s)")
    print("(In production, use the default window_seconds=60.0 for a true")
    print(" per-minute limit.)")

    print("\n=== Exercise 2: PromptTemplate ===")
    qa_template = PromptTemplate(
        template="You are a {role}. Answer this: {question}"
    )
    print("Required vars:", qa_template.required_vars)
    print(qa_template.render(role="Python tutor", question="What is a decorator?"))
    try:
        qa_template.render(role="Python tutor")   # missing "question"
    except ValueError as e:
        print(f"Raised as expected: {e}")

    print("\n=== Exercise 3: retry decorator ===")
    print(fails_twice_then_succeeds())
