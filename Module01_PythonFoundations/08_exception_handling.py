# Module 01 - Python Foundations
# 1.6 Exception Handling
# AI API calls fail. Rate limits hit, networks time out, model responses are
# malformed. Defensive exception handling is not optional - it is required for
# production code.

import time
from typing import Optional


class APIError(Exception):
    """Raised when an AI API returns an error response."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def call_api_with_retry(
    prompt: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Optional[str]:
    """Call a mock API with exponential backoff on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            # Simulate API call - replace with real client call
            if attempt < 3:
                raise APIError("Rate limit exceeded", 429)
            return f"Response to: {prompt}"
        except APIError as e:
            if e.status_code == 429 and attempt < max_retries:
                wait = backoff_seconds ** attempt
                print(f"Rate limited. Retrying in {wait:.1f}s...")
                time.sleep(0.01)   # shortened for demo
            else:
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


result = call_api_with_retry("What is a neural network?")
print(result)

# NOTE:
# Always create specific exception subclasses for your own errors (APIError,
# ValidationError). Catching bare Exception hides bugs; catching your own types
# lets you handle known failure modes gracefully.
