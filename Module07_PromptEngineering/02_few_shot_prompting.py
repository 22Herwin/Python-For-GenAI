# Module 07 - Prompt Engineering
# 7.3 Few-Shot Prompting
# Few-shot prompting provides examples of the desired input -> output mapping
# inside the prompt. It is the fastest way to teach a model an unusual format
# or task without fine-tuning.
#
# NOTE: This script makes REAL, billed API calls to Anthropic (one per test
# input). It requires a valid ANTHROPIC_API_KEY in a .env file in this folder.

import os
from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, has_anthropic_access, MODEL

load_dotenv()

FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI benchmark result string,
extract: model name, task, and score as a JSON object.

Examples:

Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score": 87.3}

Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval", "score": 92.1}

Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score": 78.9}

Return ONLY the JSON object. No explanation."""

TEST_INPUTS = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",
]


def run_few_shot_demo() -> None:
    client = get_anthropic_client()

    for text in TEST_INPUTS:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=128,
            system=FEW_SHOT_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        print(f"Input: {text}")
        print(f"Output: {resp.content[0].text}\n")


if __name__ == "__main__":
    if not has_anthropic_access():
        print("ANTHROPIC_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        run_few_shot_demo()
