# Module 02 - Data Structures & Comprehensions
# 2.5 Comprehensions
# Comprehensions are concise, readable, and fast (implemented in C). They replace
# explicit for loops when the goal is to build a new sequence from an existing one.

# ── List Comprehensions ──────────────────────────────────────────────────────
messages = [
    {"role": "user",      "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    {"role": "user",      "content": "Give an example."},
]

# Extract only user messages
user_messages = [m["content"] for m in messages if m["role"] == "user"]
print(user_messages)

# Word count per message
word_counts = [len(m["content"].split()) for m in messages]
print(word_counts)   # [3, 7, 3]

# Flatten a nested list
keywords = [["RAG", "retrieval"], ["LLM", "embedding"], ["vector"]]
flat     = [kw for group in keywords for kw in group]
print(flat)


# ── Dict and Set Comprehensions ──────────────────────────────────────────────
models           = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]
context_windows  = [128_000, 200_000, 1_000_000]

# Dict comprehension
model_context = {m: c for m, c in zip(models, context_windows)}
print(model_context)

# Filter to models with > 150K context
large_context = {m: c for m, c in model_context.items() if c > 150_000}
print(large_context)

# Set comprehension - unique word lengths
sentence = "the quick brown fox jumps over the lazy dog"
unique_lengths = {len(w) for w in sentence.split()}
print(sorted(unique_lengths))   # [2, 3, 4, 5]
