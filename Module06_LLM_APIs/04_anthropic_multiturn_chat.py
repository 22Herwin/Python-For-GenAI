# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Multi-turn conversation
# The Anthropic API is stateless - you must send the full conversation history
# on every call. Build the history yourself.
#
# NOTE: This is an interactive CLI chat loop that makes REAL, billed API calls.
# Run it directly (python 04_anthropic_multiturn_chat.py) and type "exit" to quit.

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]) if os.getenv("ANTHROPIC_API_KEY") else None


def chat(system: str) -> None:
    """Simple interactive multi-turn chat loop."""
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system,
            messages=history,
        )

        assistant_text = response.content[0].text
        history.append({"role": "assistant", "content": assistant_text})
        print(f"Claude: {assistant_text}\n")


if __name__ == "__main__":
    if client is None:
        print("ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        chat(system="You are a helpful Python tutor.")
