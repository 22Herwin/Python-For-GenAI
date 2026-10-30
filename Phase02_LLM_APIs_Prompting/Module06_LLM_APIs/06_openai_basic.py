# Module 06 - OpenAI & Anthropic APIs
# 6.3 The OpenAI SDK - Basic message call
# The openai package follows a nearly identical pattern to Anthropic's SDK.
#
# NOTE: This script makes a REAL, billed API call to OpenAI. It requires a
# valid OPENAI_API_KEY in a .env file in this folder to run.

from dotenv import load_dotenv

from _openrouter_client import get_openai_client, has_openai_access, OPENAI_MODEL

load_dotenv()


def basic_call() -> None:
    client = get_openai_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "What is the difference between RAG and fine-tuning?"},
        ],
    )

    print(response.choices[0].message.content)
    print(f"Tokens used: {response.usage.total_tokens}")


if __name__ == "__main__":
    if not has_openai_access():
        print("OPENAI_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        basic_call()
