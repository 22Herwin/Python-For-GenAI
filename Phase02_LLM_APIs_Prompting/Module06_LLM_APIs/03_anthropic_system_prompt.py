# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - System prompt
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, has_anthropic_access, MODEL

load_dotenv()


def call_with_system_prompt() -> None:
    client = get_anthropic_client()

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="You are a concise technical writer. Answer in plain English, no jargon.",
        messages=[
            {"role": "user", "content": "Explain what a vector database does."}
        ],
    )

    print(message.content[0].text)


if __name__ == "__main__":
    if not has_anthropic_access():
        print("ANTHROPIC_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        call_with_system_prompt()
