# Module 07 - Prompt Engineering
# 7.4 Chain-of-Thought Prompting
# Chain-of-thought (CoT) prompting asks the model to reason step by step before
# giving a final answer. It consistently improves accuracy on tasks requiring
# multi-step reasoning - math, logic, planning.
#
# NOTE: This script makes REAL, billed API calls to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Without CoT - model jumps to answer, more likely to be wrong
DIRECT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million "
    "output tokens, and a request uses 2,400 input tokens and 800 output "
    "tokens, what is the total cost in USD?"
)

# With CoT - model reasons through each step
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens,
what is the total cost in USD?

Think through this step by step before giving the final answer."""

# Zero-shot CoT: just adding "think step by step"
ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX

Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens
and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

# Structured CoT with XML tags - makes it easy to parse the final answer
STRUCTURED_SYSTEM = """Solve problems using this exact format:

<thinking>
Step-by-step reasoning here.
</thinking>

<answer>
The final answer only, no reasoning.
</answer>"""

STRUCTURED_QUESTION = (
    "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 "
    "tokens. The model has a 4096 token limit for context. How many tokens "
    "remain for the response?"
)


def compare_direct_vs_cot() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for label, prompt in [("Direct", DIRECT_PROMPT), ("CoT", COT_PROMPT), ("Zero-shot CoT", ZERO_SHOT_COT)]:
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"=== {label} ===")
        print(resp.content[0].text[:300])
        print()


def structured_cot_demo() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=STRUCTURED_SYSTEM,
        messages=[{"role": "user", "content": STRUCTURED_QUESTION}],
    )

    text = resp.content[0].text

    # Extract sections
    thinking = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    print("Reasoning:", thinking.group(1).strip() if thinking else "not found")
    print("Answer:   ", answer.group(1).strip() if answer else "not found")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        compare_direct_vs_cot()
        print("=" * 40)
        structured_cot_demo()
