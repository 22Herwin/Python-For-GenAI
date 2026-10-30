# Module 04 - File I/O & APIs
# 4.4 Module 04 Exercises
#
# All four exercises solved below. Exercise 2 needs internet access (it hits a
# public test API); it is wrapped in a try/except so the rest of the file still
# runs if you are offline.

import asyncio
import csv
import json
import os
import threading
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Write save_conversation(history, path) and load_conversation(path) that
# serialise and deserialise conversation history to/from a JSON file.

def save_conversation(history: list[dict], path: str) -> None:
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_conversation(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Build an async function compare_endpoints(urls) that fetches all URLs
# concurrently with httpx and returns a list of
# (url, status_code, response_time_ms) tuples.

async def compare_endpoints(urls: list[str]) -> list[tuple[str, int, float]]:
    async def fetch_one(client: httpx.AsyncClient, url: str) -> tuple[str, int, float]:
        start = time.perf_counter()
        try:
            response = await client.get(url, timeout=10.0)
            status_code = response.status_code
        except httpx.HTTPError:
            status_code = -1
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return (url, status_code, elapsed_ms)

    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Create a config loader that reads a JSON config file and merges it with
# environment variable overrides (env var values take precedence).

def load_config(config_path: str) -> dict:
    """Load JSON config, then override any key whose UPPERCASE name is also
    set as an environment variable."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key in list(config.keys()):
        env_key = key.upper()
        if env_key in os.environ:
            config[key] = os.environ[env_key]
    return config


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Write a CSV log writer class that appends a row each time an LLM is called,
# recording: timestamp, model, input_tokens, output_tokens, latency_ms. Make
# it thread-safe with a threading.Lock.

class LLMCallLogger:
    """Thread-safe append-only CSV logger for LLM API calls."""

    _FIELDNAMES = ["timestamp", "model", "input_tokens", "output_tokens", "latency_ms"]

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._FIELDNAMES).writeheader()

    def log(self, model: str, input_tokens: int, output_tokens: int, latency_ms: float) -> None:
        row = {
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model":         model,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "latency_ms":    latency_ms,
        }
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._FIELDNAMES).writerow(row)


if __name__ == "__main__":
    print("=== Exercise 1: save/load conversation ===")
    convo = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "Retrieval-Augmented Generation."},
    ]
    convo_path = DATA_DIR / "conversation.json"
    save_conversation(convo, str(convo_path))
    loaded = load_conversation(str(convo_path))
    print(loaded)
    assert loaded == convo

    print("\n=== Exercise 2: compare_endpoints (needs internet) ===")
    try:
        endpoint_urls = [
            f"https://jsonplaceholder.typicode.com/posts/{i}" for i in range(1, 4)
        ]
        results = asyncio.run(compare_endpoints(endpoint_urls))
        for url, status, ms in results:
            print(f"{url} -> status={status}, {ms}ms")
    except Exception as e:
        print(f"Skipped (no internet or request failed): {e}")

    print("\n=== Exercise 3: load_config with env overrides ===")
    config_path = DATA_DIR / "config.json"
    config_path.write_text(json.dumps({"model": "gpt-4o", "temperature": 0.7}), encoding="utf-8")
    os.environ["MODEL"] = "claude-sonnet-4-5"   # simulate an env override
    merged = load_config(str(config_path))
    print(merged)   # "model" overridden by MODEL env var, "temperature" untouched

    print("\n=== Exercise 4: LLMCallLogger (thread-safe) ===")
    log_path = DATA_DIR / "llm_calls.csv"
    if log_path.exists():
        log_path.unlink()   # start fresh for a clean demo
    logger = LLMCallLogger(str(log_path))

    def worker(worker_id: int) -> None:
        for i in range(3):
            logger.log(
                model=f"model-{worker_id}",
                input_tokens=100 + i,
                output_tokens=50 + i,
                latency_ms=round(200 + i * 10.5, 1),
            )

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with log_path.open(encoding="utf-8") as f:
        row_count = sum(1 for _ in csv.DictReader(f))
    print(f"Logged {row_count} rows from 4 threads x 3 calls each (expected 12)")
