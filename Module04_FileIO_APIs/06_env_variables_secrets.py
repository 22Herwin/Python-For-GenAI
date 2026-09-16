# Module 04 - File I/O & APIs
# 4.3 Environment Variables and Secrets
# Never hardcode API keys. Use environment variables, loaded via python-dotenv
# in development.
#
# Copy `.env.example` to `.env` in this folder and fill in real keys to make
# get_api_key() succeed. `.env` is already listed in `.gitignore` so it is
# never committed. A leaked key can run up thousands of dollars in API costs
# within hours - add .env to .gitignore before your first commit.

import os
from dotenv import load_dotenv

load_dotenv()   # loads .env into os.environ


def get_api_key(provider: str) -> str:
    """Retrieve an API key from the environment."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
    }
    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(
            f"{env_var} is not set. Add it to your .env file."
        )
    return key


if __name__ == "__main__":
    for provider in ("anthropic", "openai", "google"):
        try:
            key = get_api_key(provider)
            print(f"{provider}: found key starting with {key[:6]}...")
        except EnvironmentError as e:
            print(f"{provider}: {e}")

    try:
        get_api_key("mistral")
    except ValueError as e:
        print(f"Raised as expected: {e}")
