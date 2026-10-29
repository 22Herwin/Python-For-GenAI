# Module 13 - LangGraph & Agents
# 13.3 ReAct Agent - Reason + Act
# The ReAct pattern (Reason + Act) is the most common agent architecture. The
# model reasons about what to do, calls a tool, observes the result, and
# repeats until it has a final answer. Graph: START -> agent <-> tools -> END.
#
# NOTE: The default demo uses a scripted fake LLM (no key, no cost) so the tool
# loop can be tested offline. Pass --real to use ChatOpenAI instead - that makes
# REAL, billed API calls and needs OPENAI_API_KEY (or OPENROUTER_API_KEY /
# OMNIROUTE_API_KEY) in a .env file in this folder. Requires langgraph,
# langchain-core and, for --real, langchain-openai.

import json
import math
import operator
import sys
from typing import Annotated, Callable, TypedDict

from _openrouter_client import (
    LANGGRAPH_INSTALL_HINT,
    NO_KEY_HINT,
    get_chat_model,
    has_llm_access,
)

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ── Tools ──────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    try:
        # Allow only math-module names; builtins are removed. This keeps the
        # demo tidy but is not a hardened sandbox - never eval untrusted input
        # in production.
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def token_cost_lookup(model: str) -> str:
    """Return cost per 1M tokens for a given model."""
    costs = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    }
    if model not in costs:
        return f"Unknown model: {model}. Known: {list(costs.keys())}"
    c = costs[model]
    return json.dumps({"model": model, "input_per_1M": c["input"], "output_per_1M": c["output"]})


TOOLS: dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "token_cost_lookup": token_cost_lookup,
}

# OpenAI-style tool schemas - what the model sees via .bind_tools(...)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use Python math syntax.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "token_cost_lookup",
            "description": "Look up the cost per 1M tokens for a given LLM model.",
            "parameters": {
                "type": "object",
                "properties": {"model": {"type": "string", "description": "Model identifier"}},
                "required": ["model"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful AI assistant with access to two tools:
- calculator: for math calculations
- token_cost_lookup: for LLM pricing information

Always use tools when calculations or pricing lookups are needed. Think step by step."""


# ── State ──────────────────────────────────────────────────────────────────────

class ReActState(TypedDict):
    messages: Annotated[list, operator.add]


# ── Nodes ──────────────────────────────────────────────────────────────────────

def tool_node(state: ReActState) -> dict:
    """Execute all requested tool calls and return results."""
    last_msg = state["messages"][-1]
    results = []
    for tool_call in last_msg.tool_calls:
        fn = TOOLS.get(tool_call["name"])
        output = fn(**tool_call["args"]) if fn else f"Error: unknown tool {tool_call['name']!r}"
        results.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return {"messages": results}


def should_use_tools(state: ReActState) -> str:
    """Route to the tools node if the model asked for a tool, else finish."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_react_agent(llm):
    """Compile the ReAct graph around any LLM with .invoke(messages) -> AIMessage
    (a tool-bound ChatOpenAI, or the ScriptedChatLLM below)."""
    system = SystemMessage(content=SYSTEM_PROMPT)

    def agent_node(state: ReActState) -> dict:
        """Call the LLM - it may request tool use or give a final answer."""
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")   # after tools, back to agent

    return graph.compile()


# ── Scripted fake LLM (offline) ────────────────────────────────────────────────

class ScriptedChatLLM:
    """Fake chat model that replays a fixed script, one entry per .invoke().

    Each script entry is either an AIMessage or a callable(messages) -> AIMessage,
    so a later turn can read earlier tool results - which is how a real model
    behaves, minus the network call."""

    def __init__(self, script: list):
        self._script = list(script)
        self._turn = 0

    def bind_tools(self, *args, **kwargs) -> "ScriptedChatLLM":
        return self   # tools are scripted, nothing to bind

    def invoke(self, messages: list, *args, **kwargs):
        if self._turn >= len(self._script):
            raise RuntimeError("ScriptedChatLLM ran out of scripted turns.")
        step = self._script[self._turn]
        self._turn += 1
        return step(messages) if callable(step) else step


def _tool_outputs(messages: list) -> list[str]:
    return [m.content for m in messages if isinstance(m, ToolMessage)]


def _tool_call_msg(name: str, args: dict, call_id: str):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def build_pricing_script() -> list:
    """Three turns: look up gpt-4o pricing -> calculate daily cost -> answer."""

    def calculate_step(messages: list):
        prices = json.loads(_tool_outputs(messages)[-1])
        expression = (
            f"10000 * (800 * {prices['input_per_1M']} + 300 * {prices['output_per_1M']}) / 1_000_000"
        )
        return _tool_call_msg("calculator", {"expression": expression}, "call_2")

    def final_step(messages: list):
        daily_cost = float(_tool_outputs(messages)[-1])
        return AIMessage(content=(
            f"Your pipeline costs about ${daily_cost:,.2f} per day "
            "(10,000 requests x 800 input + 300 output tokens on gpt-4o)."
        ))

    return [
        _tool_call_msg("token_cost_lookup", {"model": "gpt-4o"}, "call_1"),
        calculate_step,
        final_step,
    ]


# ── Run ────────────────────────────────────────────────────────────────────────

QUESTION = """I'm running a pipeline that processes 10,000 requests per day.
Each request uses 800 input tokens and 300 output tokens with gpt-4o.
What is my daily cost in USD?"""


def run_agent(llm, question: str = QUESTION) -> None:
    agent = build_react_agent(llm)
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  [tool call]   {call['name']}({call['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"  [observation] {msg.content}")

    # Print the final answer
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            print("Final answer:\n", msg.content)


if __name__ == "__main__":
    if not LANGGRAPH_AVAILABLE:
        print(f"langgraph / langchain-core is not installed. Run: {LANGGRAPH_INSTALL_HINT}")
        sys.exit(0)

    print("=== ReAct agent with a scripted fake LLM (offline) ===")
    run_agent(ScriptedChatLLM(build_pricing_script()))

    print("\n=== ReAct agent with a real LLM (--real) ===")
    if "--real" not in sys.argv:
        print("Skipped - run with --real to call the real model (billed).")
    elif not has_llm_access():
        print(f"Skipped - {NO_KEY_HINT}")
    else:
        try:
            run_agent(get_chat_model().bind_tools(TOOL_SCHEMAS))
        except ImportError as exc:
            print(f"Skipped - {exc}")
