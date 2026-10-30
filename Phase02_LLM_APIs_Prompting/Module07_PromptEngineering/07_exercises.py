# Module 07 - Prompt Engineering
# 7.8 Module 07 Exercises
#
# Exercises 2 and 3 (tiers 1-2) are fully testable offline. Exercises 1 and 4,
# and tier 3 of exercise 3, need a real ANTHROPIC_API_KEY to actually call the
# model - the code is complete and correct, just gated behind a key check.

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Optional

from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, has_anthropic_access, MODEL

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Shared: a small PromptTemplate (same as section 7.6) ──────────────────────

@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self):
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [
            fname for _, fname, _, _ in formatter.parse(combined)
            if fname is not None
        ]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Write three versions of a system prompt for a "code review assistant" -
# basic, intermediate, and expert-level. Evaluate all three on the same 5
# code snippets and compare output quality.

CODE_REVIEW_BASIC = "You are a code reviewer. Review the code and point out any issues."

CODE_REVIEW_INTERMEDIATE = """You are a code reviewer. Review the given code for:
- Bugs and correctness issues
- Readability
- Basic security concerns

List each issue you find with a short explanation."""

CODE_REVIEW_EXPERT = """You are a senior software engineer performing a production code review.

Check for:
- Correctness bugs (logic errors, edge cases, off-by-one errors)
- Security vulnerabilities (injection, unsafe deserialization, secrets in code)
- Performance issues (unnecessary loops, N+1 queries, blocking calls)
- Maintainability (naming, duplication, missing error handling)

Rules:
- Be direct, no filler praise.
- For every issue: state Impact, then give the corrected code.
- If the code has no issues, say so in one sentence.

Format: numbered list, one issue per item."""

CODE_SNIPPETS = [
    "def add(a, b):\n    return a + b",
    "def get_user(id):\n    return db.execute(f'SELECT * FROM users WHERE id={id}')",
    "def divide(a, b):\n    return a / b",
    "results = []\nfor item in items:\n    for other in items:\n        results.append(item == other)",
    "password = 'admin123'\ndef login(pw):\n    return pw == password",
]


def compare_review_prompts() -> None:
    client = get_anthropic_client()
    prompts = {
        "basic": CODE_REVIEW_BASIC,
        "intermediate": CODE_REVIEW_INTERMEDIATE,
        "expert": CODE_REVIEW_EXPERT,
    }
    for label, system in prompts.items():
        print(f"\n=== {label} ===")
        for snippet in CODE_SNIPPETS[:1]:   # demo with 1 snippet to limit API cost
            resp = client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": snippet}],
            )
            print(resp.content[0].text[:300])


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Build a PromptLibrary class that stores named PromptTemplate instances,
# supports saving/loading to JSON, and tracks which version of each template
# produced the last evaluation run.

class PromptLibrary:
    """In-memory + JSON-persisted store of named PromptTemplate instances,
    with tracking of the last evaluation run per template."""

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._last_eval: dict[str, dict] = {}   # name -> {version, pass_rate, ...}

    def add(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        return self._templates[name]

    def names(self) -> list[str]:
        return list(self._templates.keys())

    def record_evaluation(self, name: str, pass_rate: float) -> None:
        """Call this after running an evaluation against a template, to track
        which version produced that result."""
        template = self._templates[name]
        self._last_eval[name] = {"version": template.version, "pass_rate": pass_rate}

    def last_evaluation(self, name: str) -> Optional[dict]:
        return self._last_eval.get(name)

    def save(self, path: str) -> None:
        data = {
            "templates": {
                name: {"name": t.name, "system": t.system, "user": t.user, "version": t.version}
                for name, t in self._templates.items()
            },
            "last_eval": self._last_eval,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "PromptLibrary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lib = cls()
        for t in data["templates"].values():
            lib.add(PromptTemplate(name=t["name"], system=t["system"], user=t["user"], version=t["version"]))
        lib._last_eval = data.get("last_eval", {})
        return lib


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Implement automatic JSON repair: safe_json_parse(text) -> dict that first
# tries json.loads, then strips markdown fences, then asks the model to fix
# the JSON if it is still invalid.

def safe_json_parse(text: str, client=None, model: str = MODEL) -> dict:
    """Tier 1: direct parse. Tier 2: strip markdown fences. Tier 3 (needs a
    client): ask the model to repair the JSON."""
    # Tier 1
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tier 2 - strip ```json ... ``` or ``` ... ``` fences
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Tier 3 - ask the model to repair it
    if client is None:
        raise ValueError("JSON could not be parsed and no client was provided for repair.")

    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system="Fix the following text so it becomes valid JSON. Return ONLY the corrected JSON, nothing else.",
        messages=[{"role": "user", "content": text}],
    )
    repaired = resp.content[0].text.strip()
    repaired = re.sub(r"^```(?:json)?\s*|\s*```$", "", repaired)
    return json.loads(repaired)   # let this raise if the model still couldn't fix it


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Design a CoT prompt: given a list of 10 LLM evaluation scores across 3
# tasks, rank the models and write a 2-sentence recommendation. Verify it
# works correctly on at least 3 different inputs.

RANKING_COT_SYSTEM = """You rank LLMs based on evaluation scores.

Think step by step:
1. List each model's average score across all tasks.
2. Rank models from highest to lowest average.
3. Identify which task each model is strongest/weakest at.

Then write exactly 2 sentences recommending which model to use and why.

Format your response as:
<reasoning>
...step-by-step reasoning...
</reasoning>
<recommendation>
...exactly 2 sentences...
</recommendation>"""

RANKING_TEST_INPUTS = [
    """Scores (0-100):
gpt-4o: qa=88, summarise=82, code=91
claude-sonnet-4-5: qa=91, summarise=89, code=93
gemini-1.5-pro: qa=85, summarise=90, code=80""",
    """Scores (0-100):
model-a: qa=70, summarise=95, code=60
model-b: qa=90, summarise=60, code=95
model-c: qa=80, summarise=80, code=80""",
    """Scores (0-100):
fast-model: qa=75, summarise=75, code=75
big-model: qa=95, summarise=95, code=95
cheap-model: qa=65, summarise=70, code=68""",
]


def run_ranking_cot_demo() -> None:
    client = get_anthropic_client()
    for i, scores_text in enumerate(RANKING_TEST_INPUTS, 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=RANKING_COT_SYSTEM,
            messages=[{"role": "user", "content": scores_text}],
        )
        text = resp.content[0].text
        rec = re.search(r"<recommendation>(.*?)</recommendation>", text, re.DOTALL)
        print(f"--- Input {i} ---")
        print("Recommendation:", rec.group(1).strip() if rec else "(not found)")
        print()


if __name__ == "__main__":
    has_key = has_anthropic_access()

    print("=== Exercise 1: compare_review_prompts (needs API key) ===")
    if has_key:
        compare_review_prompts()
    else:
        print("Skipped - ANTHROPIC_API_KEY not set.")

    print("\n=== Exercise 2: PromptLibrary (offline) ===")
    library = PromptLibrary()
    library.add(PromptTemplate(
        name="qa", version="1.0",
        system="You are a {domain} expert.",
        user="Question: {question}",
    ))
    library.record_evaluation("qa", pass_rate=0.8)
    lib_path = DATA_DIR / "prompt_library.json"
    library.save(str(lib_path))
    reloaded = PromptLibrary.load(str(lib_path))
    print("Templates:", reloaded.names())
    print("Last eval for 'qa':", reloaded.last_evaluation("qa"))

    print("\n=== Exercise 3: safe_json_parse (tiers 1-2 offline) ===")
    print(safe_json_parse('{"a": 1}'))                          # tier 1
    print(safe_json_parse('```json\n{"a": 1}\n```'))            # tier 2
    if has_key:
        client = get_anthropic_client()
        print(safe_json_parse('{"a": 1,}', client=client))      # tier 3 (trailing comma)
    else:
        print("Tier 3 (model repair) skipped - ANTHROPIC_API_KEY not set.")

    print("\n=== Exercise 4: ranking CoT prompt (needs API key) ===")
    if has_key:
        run_ranking_cot_demo()
    else:
        print("Skipped - ANTHROPIC_API_KEY not set.")
