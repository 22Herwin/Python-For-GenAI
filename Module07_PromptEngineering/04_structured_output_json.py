# Module 07 - Prompt Engineering
# 7.5 Structured Output (JSON Mode)
# Getting models to return valid, parseable JSON is one of the most common
# production requirements. There are two approaches.
#
# NOTE: Both demos make REAL, billed API calls (Anthropic and OpenAI
# respectively). They require ANTHROPIC_API_KEY / OPENAI_API_KEY in a .env
# file in this folder to run.

import json
import os

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ── Approach 1: Prompt-enforced JSON (works with any model) ──────────────────

ANTHROPIC_SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

COMPANY_TEXTS = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude "
    "AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San "
    "Francisco, it remains private despite a major Microsoft investment.",
]


def extract_company_info(text: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        system=ANTHROPIC_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = resp.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def run_anthropic_demo() -> None:
    for text in COMPANY_TEXTS:
        info = extract_company_info(text)
        print(json.dumps(info, indent=2))
        print()


# ── Approach 2: OpenAI JSON mode (response_format enforces valid JSON) ───────

def run_openai_json_mode_demo() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},   # enforces valid JSON
        messages=[
            {
                "role": "system",
                "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}""",
            },
            {
                "role": "user",
                "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla.",
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)
    print(result)
    # {'people': ['Elon Musk'], 'organizations': ['SpaceX', 'Tesla'], 'locations': ['Hawthorne, California']}


if __name__ == "__main__":
    if os.getenv("ANTHROPIC_API_KEY"):
        print("=== Approach 1: Anthropic prompt-enforced JSON ===")
        run_anthropic_demo()
    else:
        print("Skipping Anthropic demo - ANTHROPIC_API_KEY not set.")

    if os.getenv("OPENAI_API_KEY"):
        print("=== Approach 2: OpenAI json_object mode ===")
        run_openai_json_mode_demo()
    else:
        print("Skipping OpenAI demo - OPENAI_API_KEY not set.")
