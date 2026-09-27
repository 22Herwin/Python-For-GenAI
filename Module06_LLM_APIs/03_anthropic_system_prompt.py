# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - System prompt
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


def call_with_system_prompt() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system="You are a concise technical writer. Answer in plain English, no jargon.",
        messages=[
            {"role": "user", "content": "Explain what a vector database does."}
        ],
    )

    print(message.content[0].text)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        call_with_system_prompt()
