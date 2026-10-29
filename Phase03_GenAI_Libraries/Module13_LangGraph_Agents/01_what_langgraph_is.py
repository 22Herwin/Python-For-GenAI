# Module 13 - LangGraph & Agents
# 13.1 What LangGraph Is
# LangGraph is a library for building stateful, multi-step LLM applications
# modelled as directed graphs. Each node is a function that reads and writes to
# a shared state dict. Edges define control flow - including conditional
# branches and loops.
#
# NOTE: Requires the langgraph package (pip install langgraph). No API key and
# no billed calls - the demo graph below runs fully offline.

from typing import TypedDict

from _openrouter_client import LANGGRAPH_INSTALL_HINT

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ── When LangGraph is the right tool ───────────────────────────────────────────

LANGGRAPH_FIT_CRITERIA = [
    "The number of LLM calls is not fixed in advance",
    "The agent must decide which tool to call next based on results",
    "You need persistent state between steps (memory, intermediate results)",
    "You want human-in-the-loop checkpoints",
]


# ── Smallest possible graph: shared state + nodes + edges ─────────────────────

class HelloState(TypedDict):
    name: str
    greeting: str


def greet_node(state: HelloState) -> dict:
    """Reads 'name' from the shared state, writes 'greeting' back."""
    return {"greeting": f"Hello, {state['name']}!"}


def shout_node(state: HelloState) -> dict:
    """A second node sees what the first node wrote."""
    return {"greeting": state["greeting"].upper()}


def build_hello_graph():
    """START -> greet -> shout -> END"""
    graph = StateGraph(HelloState)
    graph.add_node("greet", greet_node)
    graph.add_node("shout", shout_node)

    graph.add_edge(START, "greet")
    graph.add_edge("greet", "shout")
    graph.add_edge("shout", END)
    return graph.compile()


if __name__ == "__main__":
    print("LangGraph is the right tool when:")
    for criterion in LANGGRAPH_FIT_CRITERIA:
        print(f"  - {criterion}")

    if not LANGGRAPH_AVAILABLE:
        print(f"\nlanggraph is not installed. Run: {LANGGRAPH_INSTALL_HINT}")
    else:
        print("\n=== Minimal graph: nodes share one state dict ===")
        app = build_hello_graph()
        result = app.invoke({"name": "LangGraph", "greeting": ""})
        print("Final state:", result)
