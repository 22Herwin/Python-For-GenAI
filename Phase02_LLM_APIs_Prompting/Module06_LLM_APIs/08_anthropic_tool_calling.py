# Module 06 - OpenAI & Anthropic APIs
# 6.4 Tool Calling (Function Calling) - Anthropic
# Tool calling lets the model decide when to invoke a function you define. This
# is the foundation of agents: the model requests a tool, you execute it,
# return the result, and the model continues.
#
# NOTE: This script makes REAL, billed API calls to Anthropic (two calls per
# run). It requires a valid ANTHROPIC_API_KEY in a .env file in this folder.

import json
from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, has_anthropic_access, MODEL

load_dotenv()

# 1. Define the tool schema
TOOLS = [
    {
        "name": "get_model_info",
        "description": "Returns context window size and cost per 1K tokens for a given LLM.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "The model identifier, e.g. 'gpt-4o' or 'claude-sonnet-4-5'.",
                }
            },
            "required": ["model_name"],
        },
    }
]


# 2. The actual function the tool will call
def get_model_info(model_name: str) -> dict:
    db = {
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00, "cost_output": 15.00},
        "gpt-4o":            {"context_k": 128, "cost_input": 2.50, "cost_output": 10.00},
        "gemini-1.5-pro":    {"context_k": 1000, "cost_input": 1.25, "cost_output": 5.00},
    }
    return db.get(model_name, {"error": f"Unknown model: {model_name}"})


def run_tool_call_demo() -> None:
    client = get_anthropic_client()

    # 3. First API call - model may return a tool_use block
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "How large is the context window of claude-sonnet-4-5?"}
        ],
    )

    # 4. Check if model wants to use a tool
    if response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")
        tool_name = tool_block.name
        tool_input = tool_block.input
        tool_use_id = tool_block.id

        # 5. Execute the function
        result = get_model_info(**tool_input)
        print(f"Tool called: {tool_name}({tool_input})")
        print(f"Tool result: {result}")

        # 6. Send tool result back to the model
        final = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=[
                {"role": "user", "content": "How large is the context window of claude-sonnet-4-5?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(result),
                        }
                    ],
                },
            ],
        )

        print("\nFinal answer:")
        print(final.content[0].text)


if __name__ == "__main__":
    if not has_anthropic_access():
        print("ANTHROPIC_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        run_tool_call_demo()
