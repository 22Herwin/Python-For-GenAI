# Module 14 - CrewAI
# 14.3 Tools for Agents
# A custom tool subclasses crewai.tools.BaseTool: name + description tell the
# agent what the tool does, args_schema (a Pydantic model) defines and validates
# its arguments, and _run() holds the actual logic. Tools are handed to an agent
# with Agent(tools=[...]).
#
# NOTE: Everything in this file runs offline - no API key and no LLM call are
# needed, because the tools are exercised directly through _run(). It only needs
# the crewai package (for BaseTool) and pydantic installed.

import json
from typing import Type

from pydantic import BaseModel, Field, ValidationError

from _openrouter_client import require_crewai

require_crewai()

from crewai.tools import BaseTool  # noqa: E402

# USD per 1M tokens
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}

CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "gemini-1.5-pro": 1_000_000,
}


# ── Token cost calculator ────────────────────────────────────────────────────

class TokenCostCalculatorInput(BaseModel):
    """Arguments the agent must supply when calling the Token Cost Calculator."""
    model: str = Field(description="LLM model name, e.g. 'gpt-4o'")
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")


class TokenCostCalculator(BaseTool):
    """Calculates the USD cost of one LLM API call from its token counts."""
    name: str = "Token Cost Calculator"
    description: str = "Calculate the USD cost of an LLM API call given model name and token counts."
    args_schema: Type[BaseModel] = TokenCostCalculatorInput

    def _run(self, model: str, input_tokens: int, output_tokens: int) -> str:
        if model not in PRICING:
            return f"Unknown model: {model}"
        p = PRICING[model]
        cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
        return json.dumps({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })


# ── Model context window lookup ──────────────────────────────────────────────

class ModelContextLookupInput(BaseModel):
    """Arguments the agent must supply when calling the context window lookup."""
    model: str = Field(description="LLM model name")


class ModelContextLookup(BaseTool):
    """Looks up the context window size (in tokens) of an LLM."""
    name: str = "Model Context Window Lookup"
    description: str = "Look up the context window size (in tokens) of an LLM."
    args_schema: Type[BaseModel] = ModelContextLookupInput

    def _run(self, model: str) -> str:
        tokens = CONTEXT_WINDOWS.get(model, "Unknown")
        return json.dumps({"model": model, "context_window_tokens": tokens})


cost_tool = TokenCostCalculator()
context_tool = ModelContextLookup()


def demo_run_tools_directly() -> None:
    """Call _run() directly - the same code path an agent triggers, minus the LLM."""
    print("=== Run the tools directly ===")
    print("cost:   ", cost_tool._run(model="gpt-4o", input_tokens=2400, output_tokens=800))
    print("cost:   ", cost_tool._run(model="claude-sonnet-4-5", input_tokens=2400, output_tokens=800))
    print("cost:   ", cost_tool._run(model="not-a-model", input_tokens=1, output_tokens=1))
    print("context:", context_tool._run(model="claude-opus-4-5"))
    print("context:", context_tool._run(model="not-a-model"))


def demo_args_schema() -> None:
    """The Pydantic args_schema is what CrewAI shows the LLM and validates against."""
    print("\n=== args_schema (what the agent sees) ===")
    for tool in (cost_tool, context_tool):
        schema = tool.args_schema.model_json_schema()
        fields = {name: spec.get("type") for name, spec in schema["properties"].items()}
        print(f"{tool.name}: {fields}")

    print("\n=== args_schema validation ===")
    valid = TokenCostCalculatorInput(model="gpt-4o", input_tokens="1200", output_tokens=400)
    print("coerced valid input:", valid.model_dump())
    try:
        TokenCostCalculatorInput(model="gpt-4o", input_tokens="lots", output_tokens=400)
    except ValidationError as exc:
        print("rejected invalid input:", exc.errors()[0]["loc"], "-", exc.errors()[0]["msg"])


if __name__ == "__main__":
    demo_run_tools_directly()
    demo_args_schema()
