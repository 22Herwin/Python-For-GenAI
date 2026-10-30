# Module 06 - OpenAI & Anthropic APIs
# 6.5 Vision - Images as Input
# Both APIs accept images alongside text in the same message.
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY in a .env
# file in this folder to run.

import base64
from pathlib import Path
from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, MODEL

load_dotenv()

client = get_anthropic_client()


# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": url}},
                {"type": "text", "text": "Describe what you see in this image."},
            ],
        }],
    )
    return response.content[0].text


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"  # image/png, image/jpeg, image/webp, image/gif

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {"type": "text", "text": "What is in this image?"},
            ],
        }],
    )
    return response.content[0].text


if __name__ == "__main__":
    if client is None:
        print("ANTHROPIC_API_KEY / OPENROUTER_API_KEY / OMNIROUTE_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        # Some gateways (e.g. a local OmniRoute proxy) can't reach every host -
        # this URL is confirmed reachable from all three provider paths.
        text = describe_image_url(
            "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.jpg"
        )
        print(text)
