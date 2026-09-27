# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Streaming responses
# Streaming lets you display tokens as they arrive instead of waiting for the
# full response - essential for chat UIs.
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


def stream_response() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": "List 5 use cases for vector databases."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print()  # newline after stream ends

    # Access final message and usage after stream completes
    final = stream.get_final_message()
    print(f"\nTotal tokens: {final.usage.input_tokens + final.usage.output_tokens}")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        stream_response()
