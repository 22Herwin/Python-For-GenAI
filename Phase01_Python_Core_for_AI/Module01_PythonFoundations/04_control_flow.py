# Module 01 - Python Foundations
# 1.3 Control Flow
# Python uses indentation - there are no braces. Consistent 4-space indentation
# is enforced by the community and most formatters (Black, Ruff).

# ── if / elif / else ──────────────────────────────────────────────────────────


def classify_response_length(token_count: int) -> str:
    if token_count < 100:
        return "short"
    elif token_count < 500:
        return "medium"
    elif token_count < 2000:
        return "long"
    else:
        return "very long"


print(classify_response_length(80))     # short
print(classify_response_length(350))    # medium
print(classify_response_length(3000))   # very long


# ── for Loops ──────────────────────────────────────────────────────────────────

models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]

# Basic iteration
for model in models:
    print(f"Checking: {model}")

# With index - use enumerate, not range(len(...))
for i, model in enumerate(models):
    print(f"{i + 1}. {model}")

# Iterate over key-value pairs in a dict
token_limits = {"gpt-4o": 128000, "claude-sonnet-4-5": 200000}
for model, limit in token_limits.items():
    print(f"{model}: {limit:,} tokens")


# ── while Loops ────────────────────────────────────────────────────────────────

import time

MAX_RETRIES = 3
attempt = 0

while attempt < MAX_RETRIES:
    attempt += 1
    print(f"Attempt {attempt}")
    if attempt == 2:
        print("Success!")
        break
    time.sleep(0.1)
else:
    # Runs only if loop exhausted without break
    print("All retries failed")

# NOTE:
# The for/while...else construct is unique to Python. The else block runs only
# when the loop completes normally (no break). Use it for search loops where you
# need to handle "not found".
