# Module 13 - LangGraph & Agents
# 13.4 Human-in-the-Loop with Checkpoints
# LangGraph supports pausing the graph mid-execution to get human approval or
# input. A checkpointer (MemorySaver) saves the state after every step, and
# interrupt_before=[...] stops the graph right before a named node so a human
# can inspect or change the state and then resume.
#
# NOTE: Requires the langgraph package (pip install langgraph). No API key and
# no billed calls - the whole flow runs offline.
# NOTE: The book resumes with `state["approved"] = True; app.invoke(state, config)`.
# Passing a new input to an invoke() call starts a fresh run from START, which
# would re-run the plan node instead of continuing. The documented way to resume
# a paused thread is app.update_state(config, {...}) followed by
# app.invoke(None, config), which is what this file does.

import operator
from typing import Annotated, TypedDict

from _openrouter_client import LANGGRAPH_INSTALL_HINT

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ── State ──────────────────────────────────────────────────────────────────────

class ReviewState(TypedDict):
    plan: str
    approved: bool
    result: str
    messages: Annotated[list[str], operator.add]


# ── Nodes ──────────────────────────────────────────────────────────────────────

def plan_node(state: ReviewState) -> dict:
    """Draft the plan a human will review."""
    plan = "Step 1: Query vector DB. Step 2: Summarise results. Step 3: Format response."
    return {"plan": plan, "messages": [f"Plan created: {plan}"]}


def execute_node(state: ReviewState) -> dict:
    """Run the plan - but only if a human approved it."""
    if not state["approved"]:
        return {"result": "Execution skipped - plan not approved.", "messages": ["Skipped."]}
    return {"result": "Pipeline executed successfully.", "messages": ["Executed."]}


def review_gate(state: ReviewState) -> str:
    """Interrupt here for human review."""
    # Kept from the book as a routing stub; it is not wired into the graph.
    # The actual pause is done by interrupt_before=["execute"] in compile().
    return "execute"


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_review_app(checkpointer):
    """START -> plan -> [pause for human] -> execute -> END"""
    graph = StateGraph(ReviewState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", END)

    # interrupt_before pauses the graph before the named node
    return graph.compile(checkpointer=checkpointer, interrupt_before=["execute"])


def run_review(app, thread_id: str, approve: bool) -> dict:
    """Run until the interrupt, apply the human decision, then resume."""
    config = {"configurable": {"thread_id": thread_id}}

    # Run until interrupt
    state = app.invoke(
        {"plan": "", "approved": False, "result": "", "messages": []},
        config=config,
    )
    print("Plan:", state["plan"])
    print("Paused before:", app.get_state(config).next)
    print("Waiting for human approval...")

    # Human decides, the decision is written into the checkpointed state,
    # then invoke(None) resumes the same thread where it stopped.
    print(f"Human {'approves' if approve else 'rejects'} the plan.")
    app.update_state(config, {"approved": approve})
    final = app.invoke(None, config=config)

    print("Result:", final["result"])
    print("Log:   ", final["messages"])
    return final


if __name__ == "__main__":
    if not LANGGRAPH_AVAILABLE:
        print(f"langgraph is not installed. Run: {LANGGRAPH_INSTALL_HINT}")
    else:
        # MemorySaver enables checkpointing (in-memory for dev; use SqliteSaver for prod)
        checkpointer = MemorySaver()
        app = build_review_app(checkpointer)

        print("=== Approved ===")
        run_review(app, thread_id="review-session-1", approve=True)

        print("\n=== Rejected (separate thread) ===")
        run_review(app, thread_id="review-session-2", approve=False)
