# Module 07 - Prompt Engineering
# 7.2 System Prompts
# The system prompt sets the model's persona, constraints, and output format
# for an entire conversation. Write it like an employment brief: role,
# responsibilities, rules, format.
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import os
from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, has_anthropic_access, MODEL

load_dotenv()

# Poor system prompt - vague, no constraints
WEAK_SYSTEM = "You are an AI assistant."

# Strong system prompt - explicit role, rules, format
STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.

Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list. Each item: Issue -> Impact -> Fix."""

CODE_TO_REVIEW = """Review this function:

def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()"""


def review_with(system_prompt: str) -> str:
    client = get_anthropic_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": CODE_TO_REVIEW}],
    )
    return response.content[0].text


if __name__ == "__main__":
    if not has_anthropic_access():
        print("ANTHROPIC_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        print("=== STRONG_SYSTEM review ===")
        print(review_with(STRONG_SYSTEM))
