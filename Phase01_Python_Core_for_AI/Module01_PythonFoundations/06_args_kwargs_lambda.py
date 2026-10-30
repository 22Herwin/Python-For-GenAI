# Module 01 - Python Foundations
# 1.4 Functions - *args, **kwargs, and Lambda Functions

# ── *args - variable positional arguments ─────────────────────────────────────


def log_messages(*messages: str) -> None:
    for msg in messages:
        print(f"[LOG] {msg}")


log_messages("Starting", "Loading model", "Done")


# ── **kwargs - variable keyword arguments ──────────────────────────────────────


def create_api_payload(model: str, **kwargs) -> dict:
    payload = {"model": model}
    payload.update(kwargs)
    return payload


payload = create_api_payload(
    "claude-sonnet-4-5",
    max_tokens=1024,
    temperature=0.3,
    stream=True,
)
print(payload)
# {'model': 'claude-sonnet-4-5', 'max_tokens': 1024, 'temperature': 0.3, 'stream': True}


# ── Lambda Functions ────────────────────────────────────────────────────────────

responses = [
    {"model": "gpt-4o",            "tokens": 540},
    {"model": "claude-sonnet-4-5", "tokens": 310},
    {"model": "gemini-1.5-pro",    "tokens": 820},
]

# Sort by token count ascending
sorted_responses = sorted(responses, key=lambda r: r["tokens"])
for r in sorted_responses:
    print(f"{r['model']}: {r['tokens']} tokens")
