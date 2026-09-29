# Module 06 - OpenAI & Anthropic APIs
# 6.3 The OpenAI SDK - Streaming with OpenAI
#
# NOTE: This script makes a REAL, billed API call to OpenAI. It requires a
# valid OPENAI_API_KEY in a .env file in this folder to run.

from dotenv import load_dotenv

from _openrouter_client import get_openai_client, has_openai_access, OPENAI_MODEL

load_dotenv()


def stream_response() -> None:
    client = get_openai_client()

    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=512,
        stream=True,
        messages=[{"role": "user", "content": "Explain embeddings in 3 bullet points."}],
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)

    print()


if __name__ == "__main__":
    if not has_openai_access():
        print("OPENAI_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        stream_response()
