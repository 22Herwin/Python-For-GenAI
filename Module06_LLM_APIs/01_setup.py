# Module 06 - OpenAI & Anthropic APIs
# 6.1 Setting Up
# Both Anthropic and OpenAI follow the same basic pattern: install the SDK,
# load your API key from the environment, create a client, and call a method.
# Never put keys in source code.
#
# Copy .env.example (in this folder) to .env and fill in real keys before
# running any other script in Module 06/07/08 - they all rely on this pattern.

import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # os.environ[...] raises KeyError immediately with a clear message if the
    # key is missing - fail fast instead of a confusing error deep inside an
    # SDK call.
    try:
        ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
        print("ANTHROPIC_API_KEY loaded (starts with):", ANTHROPIC_API_KEY[:8] + "...")
    except KeyError:
        print("ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")

    try:
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        print("OPENAI_API_KEY loaded (starts with):", OPENAI_API_KEY[:8] + "...")
    except KeyError:
        print("OPENAI_API_KEY is not set. Add it to a .env file in this folder.")
