# Module 13 - LangGraph & Agents
# 13.2 Core Concepts
# A graph is built from three pieces: a State (a TypedDict shared by all nodes,
# with an optional reducer per key that says how updates are merged), Nodes
# (plain Python functions returning a partial state update) and Edges
# (fixed or conditional). This example loops "think" twice, then answers.
#
# NOTE: Requires the langgraph package (pip install langgraph). No API key and
# no billed calls - everything here runs offline.

import operator
from typing import Annotated, TypedDict

from _openrouter_client import LANGGRAPH_INSTALL_HINT

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ── State ──────────────────────────────────────────────────────────────────────
# State is a TypedDict - shared across all nodes.
# Annotated with a reducer function that says how to merge updates:
# operator.add appends new messages instead of overwriting the list.

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]   # append new messages
    step_count: int
    final_answer: str


# ── Nodes ──────────────────────────────────────────────────────────────────────
# Nodes are plain Python functions.

def think_node(state: AgentState) -> dict:
    """Record one reasoning step and bump the counter."""
    thought = f"Step {state['step_count'] + 1}: analysing the problem..."
    return {
        "messages": [thought],
        "step_count": state["step_count"] + 1,
    }


def answer_node(state: AgentState) -> dict:
    """Produce the final answer."""
    return {"final_answer": "Based on my analysis, the answer is X."}


# ── Conditional edge ───────────────────────────────────────────────────────────
# Decides where to go next: keep thinking until two steps are done.

def should_continue(state: AgentState) -> str:
    if state["step_count"] < 2:
        return "think"
    return "answer"


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph():
    """START -> think -(loop while step_count < 2)-> answer -> END"""
    graph = StateGraph(AgentState)
    graph.add_node("think", think_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", should_continue, {"think": "think", "answer": "answer"})
    graph.add_edge("answer", END)

    return graph.compile()


if __name__ == "__main__":
    if not LANGGRAPH_AVAILABLE:
        print(f"langgraph is not installed. Run: {LANGGRAPH_INSTALL_HINT}")
    else:
        app = build_graph()

        result = app.invoke({"messages": [], "step_count": 0, "final_answer": ""})
        print("Messages:", result["messages"])
        print("Answer: ", result["final_answer"])
