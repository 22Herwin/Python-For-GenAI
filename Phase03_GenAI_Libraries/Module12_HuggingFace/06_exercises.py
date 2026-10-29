# Module 12 - Hugging Face Transformers
# 12.6 Module 12 Exercises
#
# Exercises 1-3 accept injectable pieces (generator functions, a tokenizer, an
# NER callable) so the core logic runs fully OFFLINE with simple fakes - no
# transformers, no downloads. Pass --real to run them against real Hugging Face
# models instead. Exercise 4 is a real LoRA fine-tuning run, so it only runs
# when asked.
#
# NOTE: --real needs `pip install transformers torch` and DOWNLOADS models from
# the Hugging Face Hub: distilgpt2 (~350 MB) + gpt2 (~550 MB) for Exercise 1,
# the gpt2 tokenizer (~1 MB) for Exercise 2, and the bert-large NER model
# (~1.3 GB) for Exercise 3.
# NOTE: Exercise 4 (--train) needs `pip install peft transformers torch
# datasets accelerate`, downloads distilgpt2 (~350 MB) and trains for real
# (20 examples x 40 epochs). Expect a few minutes on CPU, much less on a GPU.
# The adapter is written to ./data/exercise4_adapter. --eval re-checks a saved
# adapter without training again. An optional HF_TOKEN in .env is only needed
# for gated models.
#
# Usage:
#   python 06_exercises.py            offline demos of Exercises 1-3
#   python 06_exercises.py --real     Exercises 1-3 with real models
#   python 06_exercises.py --train    Exercise 4: train + verify the adapter
#   python 06_exercises.py --eval     Exercise 4: verify a previously saved adapter

import re
import sys
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Protocol

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
ADAPTER_DIR = DATA_DIR / "exercise4_adapter"


def missing_packages(packages: list[str]) -> list[str]:
    """Return the subset of `packages` that is not installed."""
    return [p for p in packages if find_spec(p) is None]


def deps_ok(packages: list[str]) -> bool:
    """Print an install hint and return False if any package is missing."""
    missing = missing_packages(packages)
    if missing:
        print(f"  Missing package(s): {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
        return False
    return True


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Build a ModelBenchmark class that runs the same 10 prompts through two
# different HuggingFace text-generation pipelines, records latency and output
# length, and prints a comparison table.

GenerateFn = Callable[[str], str]

BENCHMARK_PROMPTS = [
    "The future of AI agents is",
    "Retrieval-Augmented Generation works by",
    "A vector database is used for",
    "Fine-tuning a language model means",
    "The main advantage of running models locally is",
    "Tokenisation is the process of",
    "LoRA reduces training cost because",
    "An embedding is a",
    "Prompt engineering matters because",
    "The transformer architecture relies on",
]


@dataclass
class BenchmarkRecord:
    """One prompt run through one model."""
    model: str
    prompt: str
    latency_s: float
    output_chars: int
    output_words: int


class ModelBenchmark:
    """Run the same prompts through several text-generation callables and
    compare latency and output length. Each generator is just `prompt -> text`,
    so a real Hugging Face pipeline and an offline fake are interchangeable."""

    def __init__(self, generators: dict[str, GenerateFn]):
        if len(generators) < 2:
            raise ValueError("Provide at least two generators to compare.")
        self.generators = generators
        self.records: list[BenchmarkRecord] = []

    @classmethod
    def from_pipelines(
        cls, model_a: str, model_b: str, max_new_tokens: int = 40
    ) -> "ModelBenchmark":
        """Build a benchmark from two Hugging Face text-generation pipelines."""
        from transformers import pipeline

        def make_generator(model_name: str) -> GenerateFn:
            pipe = pipeline(task="text-generation", model=model_name, device=-1)

            def generate(prompt: str) -> str:
                out = pipe(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=pipe.tokenizer.eos_token_id,
                )
                return out[0]["generated_text"]

            return generate

        return cls({model_a: make_generator(model_a), model_b: make_generator(model_b)})

    def run(self, prompts: list[str]) -> list[BenchmarkRecord]:
        """Run every prompt through every generator and record the results."""
        self.records = []
        for prompt in prompts:
            for name, generate in self.generators.items():
                start = time.perf_counter()
                text = generate(prompt)
                latency = time.perf_counter() - start
                self.records.append(
                    BenchmarkRecord(
                        model=name,
                        prompt=prompt,
                        latency_s=latency,
                        output_chars=len(text),
                        output_words=len(text.split()),
                    )
                )
        return self.records

    def averages(self) -> dict[str, dict[str, float]]:
        """Mean latency (ms), characters and words per model."""
        result: dict[str, dict[str, float]] = {}
        for name in self.generators:
            rows = [r for r in self.records if r.model == name]
            if not rows:
                continue
            n = len(rows)
            result[name] = {
                "latency_ms": sum(r.latency_s for r in rows) / n * 1000,
                "chars": sum(r.output_chars for r in rows) / n,
                "words": sum(r.output_words for r in rows) / n,
            }
        return result

    def print_table(self) -> None:
        """Print a per-prompt comparison table plus an average row."""
        if not self.records:
            print("No records yet - call run() first.")
            return

        names = list(self.generators)
        group_width = 22
        prompts = list(dict.fromkeys(r.prompt for r in self.records))

        print(f"{'#':>2}  {'prompt':<30}" + "".join(f"  {n[:group_width]:^{group_width}}" for n in names))
        print(f"{'':>2}  {'':<30}" + f"  {'ms':>8} {'chars':>6} {'words':>6}" * len(names))
        print("-" * (34 + (group_width + 2) * len(names)))

        for i, prompt in enumerate(prompts, start=1):
            line = f"{i:>2}  {prompt[:28]:<30}"
            for name in names:
                r = next(x for x in self.records if x.model == name and x.prompt == prompt)
                line += f"  {r.latency_s * 1000:>8.1f} {r.output_chars:>6d} {r.output_words:>6d}"
            print(line)

        print("-" * (34 + (group_width + 2) * len(names)))
        avg = self.averages()
        line = f"{'':>2}  {'average':<30}"
        for name in names:
            a = avg[name]
            line += f"  {a['latency_ms']:>8.1f} {a['chars']:>6.0f} {a['words']:>6.1f}"
        print(line)


def make_fake_generator(delay_s: float, suffix: str) -> GenerateFn:
    """Offline stand-in for a pipeline: sleeps a little and appends `suffix`."""
    def generate(prompt: str) -> str:
        time.sleep(delay_s)
        return f"{prompt} {suffix}"
    return generate


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Write a TokenBudgetChecker that, given a list of texts and a model name,
# filters out any text that would exceed a configurable token limit and
# returns the remaining texts with their token counts.

class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


class WhitespaceTokenizer:
    """Offline stand-in tokenizer: one token per whitespace-separated word.
    Only approximates a real subword tokenizer, but is deterministic."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class TokenBudgetChecker:
    """Keep only the texts that fit within `max_tokens` for a given model."""

    def __init__(
        self,
        model_name: str = "gpt2",
        max_tokens: int = 512,
        tokenizer: TokenizerLike | None = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._tokenizer = tokenizer
        self.rejected: list[tuple[str, int]] = []   # filled by filter()

    @property
    def tokenizer(self) -> TokenizerLike:
        """Lazily load the real Hub tokenizer unless one was injected."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def filter(self, texts: list[str]) -> list[tuple[str, int]]:
        """Return (text, token_count) for every text within the limit, in the
        original order. Over-limit texts are stored in self.rejected."""
        kept: list[tuple[str, int]] = []
        self.rejected = []
        for text in texts:
            n_tokens = self.count(text)
            if n_tokens <= self.max_tokens:
                kept.append((text, n_tokens))
            else:
                self.rejected.append((text, n_tokens))
        return kept


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Use the NER pipeline to build a function extract_entities_from_corpus(texts:
# list[str]) -> dict that returns a dict mapping entity types (PERSON, ORG, LOC)
# to lists of unique extracted values across all texts.

NerFn = Callable[[str], list[dict]]

# CoNLL-03 tags used by the book's NER model -> the names asked for here.
ENTITY_TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}


def build_ner_pipeline() -> NerFn:
    """The real NER pipeline from section 12.2 (needs transformers + torch)."""
    from transformers import pipeline

    return pipeline(
        task="ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=-1,
    )


def extract_entities_from_corpus(
    texts: list[str], ner: NerFn | None = None
) -> dict[str, list[str]]:
    """Map PERSON / ORG / LOC to the unique entity strings found across all
    texts (first-seen order). Other tags such as MISC are ignored."""
    ner = ner if ner is not None else build_ner_pipeline()

    found: dict[str, list[str]] = {"PERSON": [], "ORG": [], "LOC": []}
    for text in texts:
        for ent in ner(text):
            entity_type = ENTITY_TYPE_MAP.get(ent["entity_group"])
            if entity_type is None:
                continue
            value = " ".join(ent["word"].split())   # normalise whitespace
            if value and value not in found[entity_type]:
                found[entity_type].append(value)
    return found


# Offline stand-in: recognises a small gazetteer, same output shape as the pipeline.
FAKE_GAZETTEER = {
    "Dario Amodei": "PER",
    "Daniela Amodei": "PER",
    "Sundar Pichai": "PER",
    "Anthropic": "ORG",
    "Google": "ORG",
    "Hugging Face": "ORG",
    "San Francisco": "LOC",
    "Mountain View": "LOC",
    "New York": "LOC",
}


def fake_ner(text: str) -> list[dict]:
    """Gazetteer-based fake of the NER pipeline (aggregation_strategy='simple')."""
    entities = []
    for phrase, group in FAKE_GAZETTEER.items():
        for match in re.finditer(re.escape(phrase), text):
            entities.append({
                "entity_group": group,
                "word": phrase,
                "score": 0.99,
                "start": match.start(),
                "end": match.end(),
            })
    return sorted(entities, key=lambda e: e["start"])


SAMPLE_CORPUS = [
    "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei.",
    "Sundar Pichai leads Google from Mountain View.",
    "Hugging Face has an office in New York, and Anthropic has one in San Francisco.",
]


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Implement the full LoRA fine-tuning loop from section 12.5 on a custom
# dataset of 20 Q&A pairs about a topic of your choice. Verify the fine-tuned
# model answers those questions correctly at inference time.
#
# Topic: Git version control. Answers are short so the tiny model can memorise
# them; padding tokens are masked out of the loss (-100) and an EOS token ends
# each answer so generation stops cleanly.

QA_PAIRS: list[tuple[str, str]] = [
    ("What is Git?", "Git is a distributed version control system."),
    ("What does git init do?", "It creates a new empty Git repository."),
    ("What does git clone do?", "It copies a remote repository to your machine."),
    ("What does git add do?", "It stages changes for the next commit."),
    ("What does git commit do?", "It records a snapshot of the staged changes."),
    ("What does git status show?", "It shows the state of the working tree and staging area."),
    ("What does git log show?", "It shows the commit history."),
    ("What does git diff show?", "It shows the differences between file versions."),
    ("What is a branch?", "A branch is a movable pointer to a commit."),
    ("What does git checkout do?", "It switches branches or restores files."),
    ("What does git merge do?", "It combines the history of two branches."),
    ("What does git rebase do?", "It replays commits on top of another base commit."),
    ("What does git pull do?", "It fetches remote changes and merges them."),
    ("What does git push do?", "It uploads local commits to a remote repository."),
    ("What does git fetch do?", "It downloads remote changes without merging them."),
    ("What is a remote?", "A remote is a copy of the repository hosted elsewhere."),
    ("What is HEAD?", "HEAD points to the commit you are currently on."),
    ("What does git stash do?", "It shelves uncommitted changes for later."),
    ("What is a merge conflict?", "A merge conflict happens when two branches edit the same lines."),
    ("What is a .gitignore file?", "It lists files that Git should not track."),
]

QA_REQUIRED_PACKAGES = ["transformers", "torch", "peft", "datasets", "accelerate"]
QA_MODEL_NAME = "distilgpt2"


def qa_training_text(question: str, answer: str, eos_token: str) -> str:
    return f"Question: {question} Answer: {answer}{eos_token}"


def qa_prompt(question: str) -> str:
    return f"Question: {question} Answer:"


def normalise_answer(text: str) -> str:
    """Lower-case and drop punctuation/extra spaces for lenient comparison."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def extract_answer(generated: str) -> str:
    """Take the text after the last 'Answer:' up to the next question/newline."""
    answer = generated.split("Answer:")[-1] if "Answer:" in generated else generated
    answer = answer.split("Question:")[0]
    return answer.split("\n")[0].strip()


def train_qa_adapter(
    pairs: list[tuple[str, str]] = QA_PAIRS, adapter_dir: Path = ADAPTER_DIR
) -> None:
    """The full LoRA loop from 12.5: base model -> LoRA -> dataset -> Trainer
    -> save adapter only."""
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # 1. Base model
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(QA_MODEL_NAME)

    # 2. LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 3. Dataset (labels = input_ids, padding positions masked with -100)
    texts = [qa_training_text(q, a, tokenizer.eos_token) for q, a in pairs]

    def tokenize(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=64,
            padding="max_length",
        )
        enc["labels"] = [
            [tok if mask == 1 else -100 for tok, mask in zip(ids, masks)]
            for ids, masks in zip(enc["input_ids"], enc["attention_mask"])
        ]
        return enc

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # 4. Training arguments - many epochs because 20 examples must be memorised
    training_args = TrainingArguments(
        output_dir=str(DATA_DIR / "exercise4_output"),
        num_train_epochs=40,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )

    # 5. Train
    trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized)
    trainer.train()

    # 6. Save the adapter only (small files, not the full model)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")


def evaluate_qa_adapter(
    pairs: list[tuple[str, str]] = QA_PAIRS, adapter_dir: Path = ADAPTER_DIR
) -> float:
    """7. Load base + adapter and check each question is answered correctly
    (greedy decoding; the expected answer must appear in the generated one).
    Returns the accuracy in [0, 1]."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(QA_MODEL_NAME)
    tuned = PeftModel.from_pretrained(base, str(adapter_dir))
    tuned.eval()

    correct = 0
    for question, expected in pairs:
        inputs = tokenizer(qa_prompt(question), return_tensors="pt")
        with torch.no_grad():
            out = tuned.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = extract_answer(generated)
        ok = normalise_answer(expected) in normalise_answer(answer)
        correct += int(ok)
        print(f"  [{'OK ' if ok else 'BAD'}] {question} -> {answer}")

    accuracy = correct / len(pairs)
    print(f"Correct: {correct}/{len(pairs)} ({accuracy:.0%})")
    return accuracy


# ── Demos ────────────────────────────────────────────────────────────────────

def demo_exercise_1(real: bool) -> None:
    print("=== Exercise 1: ModelBenchmark ===")
    if real:
        if not deps_ok(["transformers", "torch"]):
            return
        try:
            bench = ModelBenchmark.from_pipelines("distilgpt2", "gpt2")
            bench.run(BENCHMARK_PROMPTS)
        except Exception as exc:  # model download problems
            print(f"  Could not run the real benchmark: {type(exc).__name__}: {exc}")
            return
    else:
        print("  (offline, fake generators)")
        bench = ModelBenchmark({
            "fake-small": make_fake_generator(0.005, "and so on."),
            "fake-large": make_fake_generator(0.02, "and so on, with a much longer continuation text."),
        })
        bench.run(BENCHMARK_PROMPTS)
    bench.print_table()


def demo_exercise_2(real: bool) -> None:
    print("\n=== Exercise 2: TokenBudgetChecker ===")
    texts = [
        "Short text.",
        "word " * 30,
        "This one is a medium sized sentence about retrieval augmented generation.",
        "word " * 200,
    ]
    if real:
        if not deps_ok(["transformers"]):
            return
        checker = TokenBudgetChecker(model_name="gpt2", max_tokens=50)
    else:
        print("  (offline, whitespace tokenizer)")
        checker = TokenBudgetChecker(max_tokens=50, tokenizer=WhitespaceTokenizer())

    try:
        kept = checker.filter(texts)
    except OSError as exc:  # tokenizer could not be downloaded
        print(f"  Could not load the tokenizer: {exc}")
        return
    print(f"  Limit: {checker.max_tokens} tokens | kept {len(kept)} of {len(texts)}")
    for text, n_tokens in kept:
        print(f"    kept     {n_tokens:>4d} tokens | {text[:50]!r}")
    for text, n_tokens in checker.rejected:
        print(f"    rejected {n_tokens:>4d} tokens | {text[:50]!r}")


def demo_exercise_3(real: bool) -> None:
    print("\n=== Exercise 3: extract_entities_from_corpus ===")
    if real:
        if not deps_ok(["transformers", "torch"]):
            return
        try:
            entities = extract_entities_from_corpus(SAMPLE_CORPUS)
        except Exception as exc:  # model download problems
            print(f"  Could not run the NER pipeline: {type(exc).__name__}: {exc}")
            return
    else:
        print("  (offline, gazetteer fake)")
        entities = extract_entities_from_corpus(SAMPLE_CORPUS, ner=fake_ner)
    for entity_type, values in entities.items():
        print(f"  {entity_type:<7s} {values}")


def demo_exercise_4(train: bool, evaluate: bool) -> None:
    print("\n=== Exercise 4: LoRA fine-tuning on 20 Git Q&A pairs ===")
    if not train and not evaluate:
        print(f"  Skipped (real training). Run with --train to fine-tune {QA_MODEL_NAME},")
        print("  or --eval to verify an already saved adapter.")
        return
    if not deps_ok(QA_REQUIRED_PACKAGES):
        return
    try:
        if train:
            train_qa_adapter()
        if not ADAPTER_DIR.exists():
            print(f"  No adapter found at {ADAPTER_DIR}. Run with --train first.")
            return
        evaluate_qa_adapter()
    except OSError as exc:  # model could not be downloaded / found
        print(f"  Could not load '{QA_MODEL_NAME}' or the adapter: {exc}")
        print("  Check your internet connection (needed once to download the model).")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    real_mode = "--real" in args

    if args & {"--train", "--eval"}:
        demo_exercise_4(train="--train" in args, evaluate="--eval" in args)
    else:
        demo_exercise_1(real_mode)
        demo_exercise_2(real_mode)
        demo_exercise_3(real_mode)
        demo_exercise_4(train=False, evaluate=False)
