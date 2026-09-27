# Module 06 - OpenAI & Anthropic APIs
# 6.3 The OpenAI SDK - Streaming with OpenAI
#
# NOTE: This script makes a REAL, billed API call to OpenAI. It requires a
# valid OPENAI_API_KEY in a .env file in this folder to run.

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def stream_response() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    stream = client.chat.completions.create(
        model="gpt-4o",
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
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        stream_response()
