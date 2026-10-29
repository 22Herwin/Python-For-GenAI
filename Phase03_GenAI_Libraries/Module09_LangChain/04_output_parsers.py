# Module 09 - LangChain
# 9.5 Output Parsers
# Parsers turn raw model output into Python values. StrOutputParser returns
# plain text, JsonOutputParser returns a dict, and with_structured_output()
# makes the model itself return a validated Pydantic object.
#
# NOTE: The offline parser demo needs no key. The structured-output demo makes
# a REAL, billed API call and requires a valid OPENAI_API_KEY (or
# OPENROUTER_API_KEY / ANTHROPIC_API_KEY) in a .env file in this folder.
# with_structured_output() is schema-enforced on OpenAI models.

from pydantic import BaseModel, Field

from _openrouter_client import ensure_ready, get_chat_model, is_ready

# Print a clear message (instead of a traceback) when a library is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core"]):
    raise SystemExit(0)

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ── Pydantic schema for structured output ──────────────────────────────────────

class ModelComparison(BaseModel):
    """Structured answer for a model-comparison question."""

    models: list[str] = Field(description="Model names compared")
    best_for_cost: str = Field(description="Most cost-effective model")
    best_for_quality: str = Field(description="Highest quality model")
    recommendation: str = Field(description="Overall recommendation")


# ── Structured output chain ────────────────────────────────────────────────────

def build_comparison_chain(llm):
    """prompt | llm.with_structured_output(ModelComparison) - the chain's result
    is a ModelComparison instance, not a string."""
    structured_llm = llm.with_structured_output(ModelComparison)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI model comparison expert."),
        ("human", "Compare these models for a startup on a budget: {models}"),
    ])
    return prompt | structured_llm


def offline_parser_demo() -> None:
    """Show the two plain parsers on a canned model reply (no API call)."""
    reply = (
        '{"models": ["gpt-4o", "gpt-4o-mini"], "best_for_cost": "gpt-4o-mini", '
        '"best_for_quality": "gpt-4o", "recommendation": "Start with gpt-4o-mini."}'
    )

    text = StrOutputParser().invoke(AIMessage(content=reply))
    print("StrOutputParser  ->", type(text).__name__)

    data = JsonOutputParser(pydantic_object=ModelComparison).parse(reply)
    print("JsonOutputParser ->", type(data).__name__, "with keys", list(data))

    comparison = ModelComparison(**data)   # validate the dict against the schema
    print("Validated        ->", comparison.best_for_cost)


if __name__ == "__main__":
    print("=== Offline parser demo ===")
    offline_parser_demo()

    print("\n=== Structured output (real model) ===")
    if is_ready(llm="auto"):
        llm = get_chat_model("auto")
        chain = build_comparison_chain(llm)

        result: ModelComparison = chain.invoke({
            "models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"
        })

        print(f"Best for cost: {result.best_for_cost}")
        print(f"Best for quality: {result.best_for_quality}")
        print(f"Recommendation: {result.recommendation}")
    else:
        ensure_ready(llm="auto")
        print("Skipped the real-model demo.")
