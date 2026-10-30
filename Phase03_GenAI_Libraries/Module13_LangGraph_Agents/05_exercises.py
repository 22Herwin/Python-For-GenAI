# Module 13 - LangGraph & Agents
# 13.5 Module 13 Exercises
#
# Every exercise here takes an injectable LLM, so the whole graph logic (tool
# loop, step limit, plan-and-execute, checkpointed memory) runs offline against
# scripted / rule-based fake models - the same pattern used in Module 08. Pass
# --real to swap in ChatOpenAI where an exercise has a real path.
#
# NOTE: Offline runs need no key and cost nothing. --real makes REAL, billed API
# calls and needs OPENAI_API_KEY (or OPENROUTER_API_KEY) in
# a .env file in this folder. Requires langgraph and langchain-core
# (+ langchain-openai for --real).

import json
import math
import operator
import re
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
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

ToolFn = Callable[..., str]
Tools = dict[str, ToolFn]


# ── Shared helpers ─────────────────────────────────────────────────────────────

class ScriptedChatLLM:
    """Fake chat model that replays a fixed script, one entry per .invoke().

    Each entry is an AIMessage or a callable(messages) -> AIMessage, so a later
    turn can read earlier tool results like a real model would."""

    def __init__(self, script: list):
        self._script = list(script)
        self._turn = 0

    def bind_tools(self, *args, **kwargs) -> "ScriptedChatLLM":
        return self

    def invoke(self, messages, *args, **kwargs):
        if self._turn >= len(self._script):
            raise RuntimeError("ScriptedChatLLM ran out of scripted turns.")
        step = self._script[self._turn]
        self._turn += 1
        return step(messages) if callable(step) else step


def tool_call_message(name: str, args: dict, call_id: str):
    """AIMessage that asks for exactly one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def tool_outputs(messages: list) -> list[str]:
    """Contents of every ToolMessage so far, oldest first."""
    return [m.content for m in messages if isinstance(m, ToolMessage)]


def tool_schema(name: str, description: str, params: dict[str, str]) -> dict:
    """OpenAI-style tool schema (what .bind_tools(...) expects). `params` maps
    each string argument name to its description; all are required."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


def run_tool_calls(tools: Tools, ai_message) -> list:
    """Execute every tool call requested by an AIMessage -> list of ToolMessage."""
    results = []
    for call in ai_message.tool_calls:
        fn = tools.get(call["name"])
        if fn is None:
            output = f"Error: unknown tool {call['name']!r}"
        else:
            try:
                output = fn(**call["args"])
            except TypeError as e:   # model passed wrong / missing arguments
                output = f"Error: bad arguments for {call['name']}: {e}"
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return results


def print_trace(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  [tool call]   {call['name']}({str(call['args'])[:100]})")
        elif isinstance(msg, ToolMessage):
            print(f"  [observation] {msg.content[:100]}")


def calculator(expression: str) -> str:
    """Evaluate a math expression using only math-module names (demo-grade
    restriction, not a hardened sandbox)."""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def token_cost_lookup(model: str) -> str:
    """Return cost per 1M tokens for a given model (same table as 13.3)."""
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


PRICING_TOOLS: Tools = {"calculator": calculator, "token_cost_lookup": token_cost_lookup}
PRICING_SCHEMAS = [
    tool_schema("calculator", "Evaluate a mathematical expression. Use Python math syntax.",
                {"expression": "Math expression to evaluate"}),
    tool_schema("token_cost_lookup", "Look up the cost per 1M tokens for a given LLM model.",
                {"model": "Model identifier"}),
]
PRICING_SYSTEM = (
    "You are a helpful AI assistant with access to two tools:\n"
    "- calculator: for math calculations\n"
    "- token_cost_lookup: for LLM pricing information\n\n"
    "Always use tools when calculations or pricing lookups are needed. Think step by step."
)


class ReActState(TypedDict):
    messages: Annotated[list, operator.add]


def build_simple_react(llm, tools: Tools, system_prompt: str):
    """The plain ReAct loop from 13.3: START -> agent <-> tools -> END."""
    system = SystemMessage(content=system_prompt)

    def agent_node(state: ReActState) -> dict:
        return {"messages": [llm.invoke([system] + state["messages"])]}

    def tool_node(state: ReActState) -> dict:
        return {"messages": run_tool_calls(tools, state["messages"][-1])}

    def should_use_tools(state: ReActState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── Exercise 1 ─────────────────────────────────────────────────────────────────
# Build a research agent using LangGraph with three tools: web_search (mock it
# to return fixed results), summarise(text) (calls an LLM), and
# save_note(content) (appends to a list). The agent should research a topic,
# summarise findings, and save a structured note.

MOCK_SEARCH_RESULTS = [
    ("LangGraph models agents as stateful graphs of nodes and edges, with built-in checkpointing.",
     "https://example.com/langgraph-overview"),
    ("The ReAct pattern alternates between reasoning, tool calls and observations until a final answer.",
     "https://example.com/react-pattern"),
    ("Human-in-the-loop breakpoints let a person approve a step before the graph continues.",
     "https://example.com/human-in-the-loop"),
]

_SOURCE_SUFFIX = re.compile(r"\s*\(source:[^)]*\)")


def offline_summarise(text: str) -> str:
    """Deterministic stand-in for an LLM summary: join the first two snippets."""
    snippets = [_SOURCE_SUFFIX.sub("", line).lstrip("- ").strip() for line in text.splitlines() if line.strip()]
    return " ".join(snippets[:2])


def llm_summariser(llm) -> Callable[[str], str]:
    """Summariser backed by a real chat model."""
    def summarise_with_llm(text: str) -> str:
        reply = llm.invoke(f"Summarise the following findings in at most two sentences:\n\n{text}")
        return reply.content
    return summarise_with_llm


class ResearchToolkit:
    """The three research tools. `notes` collects everything save_note() stores."""

    def __init__(self, summarise_fn: Callable[[str], str] = offline_summarise):
        self.notes: list[str] = []
        self._summarise_fn = summarise_fn

    def web_search(self, query: str) -> str:
        """Mock search: returns the same fixed results for any query."""
        return "\n".join(f"- {snippet} (source: {url})" for snippet, url in MOCK_SEARCH_RESULTS)

    def summarise(self, text: str) -> str:
        """Summarise text (LLM-backed when a real summariser is injected)."""
        return self._summarise_fn(text)

    def save_note(self, content: str) -> str:
        """Append a note to the notebook."""
        self.notes.append(content)
        return f"Note saved (total notes: {len(self.notes)})."

    def as_tools(self) -> Tools:
        return {"web_search": self.web_search, "summarise": self.summarise, "save_note": self.save_note}


RESEARCH_SCHEMAS = [
    tool_schema("web_search", "Search the web for a query and return result snippets with sources.",
                {"query": "Search query"}),
    tool_schema("summarise", "Summarise a block of text into a short paragraph.",
                {"text": "Text to summarise"}),
    tool_schema("save_note", "Save a structured note. Pass a JSON string with the keys "
                "'topic', 'summary' and 'sources' (list of URLs).",
                {"content": "JSON string of the note"}),
]
RESEARCH_SYSTEM = (
    "You are a research agent. For the given topic: (1) call web_search, "
    "(2) call summarise on the findings, (3) call save_note with a JSON string "
    "containing 'topic', 'summary' and 'sources'. Then reply with a one-line confirmation."
)


def build_research_script(topic: str) -> list:
    """Scripted turns: search -> summarise -> save_note -> final reply."""

    def summarise_step(messages: list):
        return tool_call_message("summarise", {"text": tool_outputs(messages)[-1]}, "r2")

    def save_step(messages: list):
        search_output, summary = tool_outputs(messages)[-2:]
        sources = re.findall(r"\(source: ([^)]+)\)", search_output)
        note = json.dumps({"topic": topic, "summary": summary, "sources": sources})
        return tool_call_message("save_note", {"content": note}, "r3")

    return [
        tool_call_message("web_search", {"query": topic}, "r1"),
        summarise_step,
        save_step,
        AIMessage(content=f"Done - saved a structured note about {topic!r}."),
    ]


def run_research_agent(llm, toolkit: ResearchToolkit, topic: str) -> dict:
    """Run the research agent and return the final graph state."""
    app = build_simple_react(llm, toolkit.as_tools(), RESEARCH_SYSTEM)
    prompt = f"Research this topic and save a structured note: {topic}"
    return app.invoke({"messages": [HumanMessage(content=prompt)]})


# ── Exercise 2 ─────────────────────────────────────────────────────────────────
# Add a maximum step limit to the ReAct agent from section 13.3: if the agent
# takes more than 8 tool calls without reaching a final answer, it should stop
# and return "Reached step limit."

STEP_LIMIT_MESSAGE = "Reached step limit."


class LimitedReActState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_calls_made: int     # tool calls executed so far
    final_answer: str


def build_limited_react(llm, tools: Tools, system_prompt: str, max_tool_calls: int = 8):
    """ReAct loop that refuses to run more than `max_tool_calls` tool calls.

    After each agent turn the router checks: executed + requested > max ->
    go to the 'limit' node (which ends the run with STEP_LIMIT_MESSAGE)
    instead of executing the tools."""
    system = SystemMessage(content=system_prompt)

    def agent_node(state: LimitedReActState) -> dict:
        response = llm.invoke([system] + state["messages"])
        update: dict = {"messages": [response]}
        if not getattr(response, "tool_calls", None):
            update["final_answer"] = response.content
        return update

    def tool_node(state: LimitedReActState) -> dict:
        last = state["messages"][-1]
        return {
            "messages": run_tool_calls(tools, last),
            "tool_calls_made": state.get("tool_calls_made", 0) + len(last.tool_calls),
        }

    def limit_node(state: LimitedReActState) -> dict:
        return {"messages": [AIMessage(content=STEP_LIMIT_MESSAGE)], "final_answer": STEP_LIMIT_MESSAGE}

    def route_after_agent(state: LimitedReActState) -> str:
        last = state["messages"][-1]
        requested = len(getattr(last, "tool_calls", None) or [])
        if requested == 0:
            return END
        if state.get("tool_calls_made", 0) + requested > max_tool_calls:
            return "limit"
        return "tools"

    graph = StateGraph(LimitedReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("limit", limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "limit": "limit", END: END})
    graph.add_edge("tools", "agent")
    graph.add_edge("limit", END)
    return graph.compile()


def run_limited_agent(llm, question: str, tools: Tools = PRICING_TOOLS,
                      system_prompt: str = PRICING_SYSTEM, max_tool_calls: int = 8) -> dict:
    app = build_limited_react(llm, tools, system_prompt, max_tool_calls)
    return app.invoke(
        {"messages": [HumanMessage(content=question)], "tool_calls_made": 0, "final_answer": ""},
        # Each tool call costs two graph steps (agent + tools); leave headroom.
        config={"recursion_limit": 2 * max_tool_calls + 10},
    )


class EndlessToolCallLLM:
    """Fake model that never gives a final answer - it keeps calling the calculator."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, *args, **kwargs) -> "EndlessToolCallLLM":
        return self

    def invoke(self, messages, *args, **kwargs):
        self.calls += 1
        return tool_call_message("calculator", {"expression": f"{self.calls} + 1"}, f"loop_{self.calls}")


# ── Exercise 3 ─────────────────────────────────────────────────────────────────
# Implement a PlanAndExecute agent with two nodes: planner (LLM generates a
# numbered list of steps) and executor (iterates through steps, calling tools
# for each). State should track the current step index.

class PlanExecuteState(TypedDict):
    task: str
    plan: list[str]
    current_step: int                          # index of the next step to run
    results: Annotated[list[str], operator.add]   # raw tool output per step
    final_answer: str


# A selector picks the tool for one step: (step_text, previous_results) -> (tool_name, args)
ToolSelector = Callable[[str, list[str]], tuple[str, dict]]

PLANNER_PROMPT = """Break the task into a short numbered list of concrete steps.
Return ONLY the list, one step per line, formatted as "1. <step>".
Every step must be doable with one of these tools: {tool_names}.

Task: {task}"""

_NUMBERED_STEP = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$", re.MULTILINE)


def parse_numbered_steps(text: str) -> list[str]:
    """Extract step texts from a '1. ...' / '2) ...' numbered list."""
    return _NUMBERED_STEP.findall(text)


def keyword_tool_selector(step: str, previous: list[str]) -> tuple[str, dict]:
    """Rule-based offline stand-in for an LLM tool picker. It reads earlier
    results (prices, partial costs) the way an LLM would read its context."""
    text = step.lower()
    if "look up" in text or "price" in text:
        return "token_cost_lookup", {"model": "gpt-4o"}
    if "input" in text:
        price = json.loads(previous[0])["input_per_1M"]
        return "calculator", {"expression": f"10000 * 800 * {price} / 1_000_000"}
    if "output" in text:
        price = json.loads(previous[0])["output_per_1M"]
        return "calculator", {"expression": f"10000 * 300 * {price} / 1_000_000"}
    if "total" in text or "add" in text:
        return "calculator", {"expression": f"{previous[1]} + {previous[2]}"}
    return "none", {}


def llm_tool_selector(llm, tool_help: str) -> ToolSelector:
    """Tool picker backed by a real chat model that answers with JSON."""

    def select(step: str, previous: list[str]) -> tuple[str, dict]:
        prompt = (
            f"Pick one tool for this step.\nTools:\n{tool_help}\n\n"
            f"Earlier results: {previous}\nStep: {step}\n\n"
            'Reply with JSON only: {"tool": "<name>", "args": {...}}'
        )
        content = llm.invoke(prompt).content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return "none", {}
        try:
            choice = json.loads(match.group(0))
            return str(choice["tool"]), dict(choice.get("args", {}))
        except (ValueError, KeyError, TypeError):
            return "none", {}

    return select


class PlanAndExecuteAgent:
    """planner -> executor (loops once per step) -> END.

    `current_step` in the state is the index of the next step to run; the
    executor runs plan[current_step], advances it, and the conditional edge
    loops back until every step is done."""

    def __init__(self, planner_llm, tools: Tools, select_tool: ToolSelector = keyword_tool_selector):
        self.planner_llm = planner_llm
        self.tools = tools
        self.select_tool = select_tool
        self.app = self._build()

    def _build(self):
        def planner_node(state: PlanExecuteState) -> dict:
            prompt = PLANNER_PROMPT.format(tool_names=", ".join(self.tools), task=state["task"])
            steps = parse_numbered_steps(self.planner_llm.invoke(prompt).content)
            return {"plan": steps or [state["task"]], "current_step": 0}

        def executor_node(state: PlanExecuteState) -> dict:
            idx = state["current_step"]
            step = state["plan"][idx]
            tool_name, args = self.select_tool(step, state["results"])
            tool = self.tools.get(tool_name)
            if tool is None:
                output = f"No tool run for this step (selector chose {tool_name!r})."
            else:
                output = tool(**args)
            update: dict = {"results": [output], "current_step": idx + 1}
            if idx + 1 >= len(state["plan"]):
                update["final_answer"] = output   # the last step's output is the answer
            return update

        def next_or_finish(state: PlanExecuteState) -> str:
            return "executor" if state["current_step"] < len(state["plan"]) else END

        graph = StateGraph(PlanExecuteState)
        graph.add_node("planner", planner_node)
        graph.add_node("executor", executor_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "executor")
        graph.add_conditional_edges("executor", next_or_finish, {"executor": "executor", END: END})
        return graph.compile()

    def run(self, task: str) -> dict:
        return self.app.invoke(
            {"task": task, "plan": [], "current_step": 0, "results": [], "final_answer": ""}
        )


PRICING_TASK = (
    "Estimate the daily cost in USD of 10,000 requests per day, each with 800 input "
    "and 300 output tokens, on gpt-4o."
)
SCRIPTED_PLAN = """1. Look up the price of gpt-4o per 1M tokens.
2. Calculate the daily input token cost.
3. Calculate the daily output token cost.
4. Add both costs to get the total daily cost."""


# ── Exercise 4 ─────────────────────────────────────────────────────────────────
# Build a multi-turn agent that maintains memory across separate invocations
# using MemorySaver and a fixed thread_id. Demonstrate that the agent remembers
# context from a previous conversation in a new one.

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]


def build_memory_agent(llm, checkpointer):
    """Single-node chat graph. The graph itself is stateless: all history
    lives in the checkpointer, keyed by thread_id."""

    def chat_node(state: ChatState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)


_NAME_RE = re.compile(r"my name is (\w+)", re.IGNORECASE)
_TOPIC_RE = re.compile(r"i(?:'m| am) learning ([^.?!]+)", re.IGNORECASE)


class MemoryAwareMockLLM:
    """Offline stand-in for a chat model. It is stateless: it can only know
    what is inside the `messages` it is handed - so if it answers correctly in
    a later turn, that history must have come from the checkpointer."""

    def invoke(self, messages, *args, **kwargs):
        humans = [m.content for m in messages if isinstance(m, HumanMessage)]
        latest = humans[-1]
        history = " ".join(humans)

        if not latest.strip().endswith("?"):
            if m := _NAME_RE.search(latest):
                return AIMessage(content=f"Nice to meet you, {m.group(1).title()}.")
            if m := _TOPIC_RE.search(latest):
                return AIMessage(content=f"Great, {m.group(1).strip()} is a good thing to learn.")
            return AIMessage(content="Noted.")

        parts = []
        if "name" in latest.lower():
            m = _NAME_RE.search(history)
            parts.append(f"Your name is {m.group(1).title()}." if m else "I don't know your name yet.")
        if "learning" in latest.lower():
            m = _TOPIC_RE.search(history)
            parts.append(f"You are learning {m.group(1).strip()}." if m else "I don't know what you are learning yet.")
        return AIMessage(content=" ".join(parts) or "I don't know.")


def chat(app, thread_id: str, text: str) -> str:
    """Send one user message on a thread; return the assistant reply."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    reply = result["messages"][-1].content
    print(f"  [{thread_id}] user: {text}\n  [{thread_id}] bot:  {reply}")
    return reply


def demo_memory(llm) -> None:
    saver = MemorySaver()
    thread = "alice-chat"

    print("-- Conversation 1 (first invocations) --")
    app_one = build_memory_agent(llm, saver)
    chat(app_one, thread, "Hi, my name is Alice.")
    chat(app_one, thread, "I am learning LangGraph.")

    print("\n-- Conversation 2: a brand-new graph object, same saver + same thread_id --")
    app_two = build_memory_agent(llm, saver)
    chat(app_two, thread, "What is my name and what am I learning?")

    print("\n-- Different thread_id: no shared memory --")
    chat(app_two, "bob-chat", "What is my name?")

    history = app_two.get_state({"configurable": {"thread_id": thread}}).values["messages"]
    print(f"\nMessages stored for {thread!r}: {len(history)}")


if __name__ == "__main__":
    if not LANGGRAPH_AVAILABLE:
        print(f"langgraph / langchain-core is not installed. Run: {LANGGRAPH_INSTALL_HINT}")
        sys.exit(0)

    use_real = "--real" in sys.argv
    if use_real and not has_llm_access():
        print(f"--real ignored - {NO_KEY_HINT}\n")
        use_real = False

    topic = "LangGraph agents"
    print("=== Exercise 1: research agent (web_search / summarise / save_note) ===")
    toolkit = ResearchToolkit()
    ex1_llm = ScriptedChatLLM(build_research_script(topic))
    if use_real:
        real_llm = get_chat_model()
        toolkit = ResearchToolkit(summarise_fn=llm_summariser(real_llm))
        ex1_llm = real_llm.bind_tools(RESEARCH_SCHEMAS)
    ex1_result = run_research_agent(ex1_llm, toolkit, topic)
    print_trace(ex1_result["messages"])
    print("Saved notes:")
    for note in toolkit.notes:
        print("  ", note)

    print("\n=== Exercise 2: ReAct agent with a step limit ===")
    runaway = run_limited_agent(EndlessToolCallLLM(), "Keep calculating forever.", max_tool_calls=8)
    print(f"Runaway agent  -> {runaway['final_answer']!r} after {runaway['tool_calls_made']} tool calls")
    short_script = [
        tool_call_message("calculator", {"expression": "2 * 21"}, "c1"),
        lambda msgs: AIMessage(content=f"The answer is {tool_outputs(msgs)[-1]}."),
    ]
    normal = run_limited_agent(ScriptedChatLLM(short_script), "What is 2 * 21?", max_tool_calls=8)
    print(f"Normal agent   -> {normal['final_answer']!r} after {normal['tool_calls_made']} tool calls")
    if use_real:
        real_run = run_limited_agent(get_chat_model().bind_tools(PRICING_SCHEMAS),
                                     "What does 1M gpt-4o input plus 1M output tokens cost in USD?")
        print(f"Real agent     -> {real_run['final_answer']!r} after {real_run['tool_calls_made']} tool calls")

    print("\n=== Exercise 3: PlanAndExecute agent ===")
    planner_llm = ScriptedChatLLM([AIMessage(content=SCRIPTED_PLAN)])
    selector: ToolSelector = keyword_tool_selector
    if use_real:
        planner_llm = get_chat_model()
        selector = llm_tool_selector(
            planner_llm,
            'calculator(expression: str) - evaluate a math expression\n'
            'token_cost_lookup(model: str) - price per 1M tokens for an LLM',
        )
    agent = PlanAndExecuteAgent(planner_llm, PRICING_TOOLS, selector)
    ex3_state = agent.run(PRICING_TASK)
    for i, (step, output) in enumerate(zip(ex3_state["plan"], ex3_state["results"]), start=1):
        print(f"  {i}. {step}\n     -> {output}")
    print(f"current_step={ex3_state['current_step']} of {len(ex3_state['plan'])}")
    print("Final answer:", ex3_state["final_answer"])

    print("\n=== Exercise 4: multi-turn memory (MemorySaver + thread_id) ===")
    demo_memory(get_chat_model() if use_real else MemoryAwareMockLLM())
