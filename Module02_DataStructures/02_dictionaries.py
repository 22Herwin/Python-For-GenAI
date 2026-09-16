# Module 02 - Data Structures & Comprehensions
# 2.2 Dictionaries
# Dictionaries are the workhorse of Python AI code. API requests, model configs,
# retrieval results, and metadata are all dicts. Python 3.7+ guarantees insertion
# order.

# Building an LLM API payload
payload = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "temperature": 0.7,
    "messages": [
        {"role": "user", "content": "What is attention in transformers?"}
    ],
}

# Access
print(payload["model"])              # claude-sonnet-4-5
print(payload.get("top_p", 1.0))     # default 1.0 if key missing

# Update
payload["temperature"] = 0.3
payload.update({"stream": True, "top_k": 40})

# Iteration
for key, value in payload.items():
    if not isinstance(value, list):
        print(f"{key}: {value}")

# Membership check
print("stream" in payload)   # True
print("top_p" in payload)    # False


# ── Nested Dicts and Default Values ─────────────────────────────────────────
from collections import defaultdict

# Track token usage per model
usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

usage["claude-sonnet-4-5"]["input"] += 350
usage["claude-sonnet-4-5"]["output"] += 210
usage["gpt-4o"]["input"] += 420
usage["gpt-4o"]["output"] += 180

for model, counts in usage.items():
    total = counts["input"] + counts["output"]
    print(f"{model}: {total} total tokens")
