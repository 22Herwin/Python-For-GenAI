# Module 02 - Data Structures & Comprehensions
# 2.7 Module 02 Exercises
#
# Fill in each function below. Run this file directly to test your solutions
# against the sample data provided.


def filter_fast_responses(responses: list[dict]) -> list[dict]:
    """
    Exercise 1:
    Given a list of API response dicts (each with "tokens" and "latency_ms" keys),
    write a comprehension that returns only responses where latency is under
    500ms, sorted by token count ascending.
    """
    fast = [r for r in responses if r["latency_ms"] < 500]
    return sorted(fast, key=lambda r: r["tokens"])


def conversation_stats(messages: list[dict]) -> dict:
    """
    Exercise 2:
    Build a function conversation_stats(messages: list[dict]) -> dict that returns
    a dict with keys "total_messages", "user_turns", "assistant_turns", and
    "avg_words_per_message".
    """
    total = len(messages)
    user_turns = sum(1 for m in messages if m["role"] == "user")
    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    word_counts = [len(m["content"].split()) for m in messages]
    avg_words = sum(word_counts) / total if total else 0.0

    return {
        "total_messages": total,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "avg_words_per_message": round(avg_words, 2),
    }


def batch_items(items, batch_size: int):
    """
    Exercise 3:
    Write a generator batch_items(items, batch_size) that yields lists of
    batch_size items from any iterable. Handle the last partial batch correctly.
    """
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # yield the last partial batch, if any
        yield batch


def fast_and_cheap_models(fast_models: list[str], cheap_models: list[str]) -> set:
    """
    Exercise 4:
    Use set operations to find models that appear in both a fast_models list and
    a cheap_models list - models that are both fast and cheap.
    """
    return set(fast_models) & set(cheap_models)


if __name__ == "__main__":
    # Sample data you can use to test your implementations above.

    sample_responses = [
        {"tokens": 120, "latency_ms": 320},
        {"tokens": 450, "latency_ms": 610},
        {"tokens": 80,  "latency_ms": 210},
        {"tokens": 300, "latency_ms": 480},
    ]

    sample_messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
        {"role": "user", "content": "Give an example."},
        {"role": "assistant", "content": "Sure, here is one."},
    ]

    sample_items = list(range(1, 11))

    sample_fast_models = ["gpt-4o-mini", "claude-haiku-4-5", "gemini-1.5-flash"]
    sample_cheap_models = ["gpt-4o-mini", "claude-haiku-4-5", "llama-3.1-8b"]

    print(filter_fast_responses(sample_responses))
    print(conversation_stats(sample_messages))
    print(list(batch_items(sample_items, batch_size=3)))
    print(fast_and_cheap_models(sample_fast_models, sample_cheap_models))
