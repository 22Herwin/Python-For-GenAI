# Module 12 - Hugging Face Transformers
# 12.2 The Pipeline API
# The pipeline function is the fastest path to inference. It handles
# tokenisation, inference, and decoding automatically. This file covers text
# classification, named entity recognition, text generation and summarisation.
#
# NOTE: Needs `pip install transformers torch`. The first run of each demo
# DOWNLOADS its model from the Hugging Face Hub (internet required), then caches
# it locally: distilbert (~270 MB), bert-large NER (~1.3 GB), distilgpt2
# (~350 MB) and bart-large-cnn (~1.6 GB). Everything runs on CPU (device=-1);
# use device=0 for the first GPU. Summarisation is the heaviest demo, so it only
# runs when you ask for it:  python 02_pipeline_api.py summarisation
# (or `all`). An optional HF_TOKEN in .env is only needed for gated models.
# NOTE: The book targets a very recent transformers release. If your installed
# version no longer ships a task used here (e.g. "summarization"), that demo
# reports the error and the others still run.

import sys
from importlib.util import find_spec
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

REQUIRED_PACKAGES = ["transformers", "torch"]


def missing_packages(packages: list[str]) -> list[str]:
    """Return the subset of `packages` that is not installed."""
    return [p for p in packages if find_spec(p) is None]


# ── Text classification (sentiment analysis) ──────────────────────────────────

SENTIMENT_TEXTS = [
    "This LLM API is incredibly fast and easy to use.",
    "The documentation is confusing and the latency is terrible.",
    "The model works fine for most tasks.",
]


def classify_sentiment(texts: list[str] = SENTIMENT_TEXTS) -> None:
    """Run sentiment analysis on a list of texts and print label + score."""
    from transformers import pipeline

    classifier = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # -1 = CPU; 0 = first GPU
    )

    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"[{result['label']} | {result['score']:.3f}] {text[:60]}")


# ── Named entity recognition ──────────────────────────────────────────────────

NER_TEXT = "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei."


def recognise_entities(text: str = NER_TEXT) -> None:
    """Extract named entities and print the span, its group and the score."""
    from transformers import pipeline

    ner = pipeline(
        task="ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=-1,
    )

    entities = ner(text)
    for ent in entities:
        print(f"{ent['word']!r:25s} {ent['entity_group']} ({ent['score']:.3f})")


# ── Text generation (small local model) ───────────────────────────────────────

def generate_text(prompt: str = "The future of AI agents is") -> None:
    """Sample two continuations of `prompt` from a small GPT-2 variant."""
    from transformers import pipeline

    generator = pipeline(
        task="text-generation",
        model="distilgpt2",  # small model - runs on CPU
        device=-1,
    )

    output = generator(
        prompt,
        max_new_tokens=50,
        num_return_sequences=2,
        temperature=0.8,
        do_sample=True,
        pad_token_id=50256,  # suppress warning for GPT-2
    )

    for i, seq in enumerate(output):
        print(f"Sequence {i + 1}: {seq['generated_text']}")


# ── Summarisation ─────────────────────────────────────────────────────────────

ARTICLE = """
Large language models have fundamentally changed how software is built.
Developers now reach for AI APIs as naturally as they reach for database clients.
The shift from rule-based systems to neural language models has accelerated
since the release of GPT-3 in 2020, and the pace has only increased with
instruction-tuned models like ChatGPT, Claude, and Gemini. These models can
write code, explain concepts, extract information, and reason through complex
problems - tasks that previously required specialised rule-based systems or
significant human effort.
"""


def summarise(article: str = ARTICLE) -> None:
    """Summarise `article` with BART (a ~1.6 GB download on first run)."""
    from transformers import pipeline

    summariser = pipeline(
        task="summarization",
        model="facebook/bart-large-cnn",
        device=-1,
    )

    summary = summariser(article, max_length=60, min_length=20, do_sample=False)
    print(summary[0]["summary_text"])


# ── Demo runner ───────────────────────────────────────────────────────────────

DEMOS: dict[str, Callable[[], None]] = {
    "classification": classify_sentiment,
    "ner": recognise_entities,
    "generation": generate_text,
    "summarisation": summarise,
}
DEFAULT_DEMOS = ["classification", "ner", "generation"]  # skips the 1.6 GB BART


def run_demo(name: str) -> None:
    """Run one demo; a failure (offline, missing task, ...) is reported, not raised."""
    print(f"\n=== {name} ===")
    try:
        DEMOS[name]()
    except Exception as exc:  # model download / task availability problems
        print(f"Demo '{name}' failed: {type(exc).__name__}: {exc}")
        print("Check your internet connection (first run downloads the model) "
              "and your transformers version.")


if __name__ == "__main__":
    missing = missing_packages(REQUIRED_PACKAGES)
    if missing:
        print(f"Missing package(s): {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
        sys.exit(0)

    requested = sys.argv[1:] or DEFAULT_DEMOS
    if "all" in requested:
        requested = list(DEMOS)

    unknown = [name for name in requested if name not in DEMOS]
    if unknown:
        print(f"Unknown demo(s): {', '.join(unknown)}. Choose from: {', '.join(DEMOS)}, all")
        sys.exit(0)

    for demo_name in requested:
        run_demo(demo_name)
