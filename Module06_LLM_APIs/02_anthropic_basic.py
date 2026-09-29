# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Basic message call
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, has_anthropic_access, MODEL

load_dotenv()


def basic_call() -> None:
    client = get_anthropic_client()

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is retrieval-augmented generation?"}
        ],
    )

    # Response text is in the first content block
    print(message.content[0].text)

    # Usage stats
    print(f"Input tokens: {message.usage.input_tokens}")
    print(f"Output tokens: {message.usage.output_tokens}")


if __name__ == "__main__":
    if not has_anthropic_access():
        print("ANTHROPIC_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        basic_call()
