# Module 06 - OpenAI & Anthropic APIs
# 6.4 Tool Calling (Function Calling) - OpenAI
#
# NOTE: This script makes REAL, billed API calls to OpenAI (two calls per
# run). It requires a valid OPENAI_API_KEY in a .env file in this folder.

import json
from dotenv import load_dotenv

from _openrouter_client import get_openai_client, has_openai_access, OPENAI_MODEL

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"],
            },
        },
    }
]


def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o":            {"context_k": 128, "cost_input": 2.50},
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name, {"error": "unknown model"})


def run_tool_call_demo() -> None:
    client = get_openai_client()

    messages = [{"role": "user", "content": "What is gpt-4o's context window?"}]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = get_model_info(**args)

        # Append assistant message and tool result
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

        final = client.chat.completions.create(model=OPENAI_MODEL, max_tokens=1024, messages=messages)
        print(final.choices[0].message.content)


if __name__ == "__main__":
    if not has_openai_access():
        print("OPENAI_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        run_tool_call_demo()
