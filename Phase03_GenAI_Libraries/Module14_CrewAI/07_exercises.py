# Module 14 - CrewAI
# 14.7 Module 14 Exercises
#
# Usage:
#   python 07_exercises.py          # offline checks only (no key, no cost)
#   python 07_exercises.py 1 3      # also run exercises 1 and 3 against the LLM
#   python 07_exercises.py all      # run all four
#
# NOTE: Exercises 1, 2, 3 and 4 call crew.kickoff(), which makes REAL, billed
# API calls (several per task; exercise 4 runs two full crews, one of them
# hierarchical). They need OPENAI_API_KEY (or OPENROUTER_API_KEY /
# ANTHROPIC_API_KEY) in a .env file in this folder, and only
# run when you name them on the command line. The offline checks always run:
# they exercise the WebSearchTool via _run() and build every crew with a
# placeholder LLM, with no network access.

import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel, Field, ValidationError

from _openrouter_client import (
    NO_KEY_MESSAGE, describe_llm, get_llm, has_llm_access, require_crewai,
)

require_crewai()

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import BaseTool  # noqa: E402


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Build a three-agent content pipeline for technical blog posts: Researcher
# (finds key points), Outline Writer (structures the post), Full Writer (expands
# each section). Run it on "The practical guide to fine-tuning vs RAG" and print
# the final post.

BLOG_TOPIC = "The practical guide to fine-tuning vs RAG"


def build_blog_crew(llm: Any, topic: str = BLOG_TOPIC) -> Crew:
    """Researcher -> Outline Writer -> Full Writer, chained with task context."""
    researcher = Agent(
        role="Blog Researcher",
        goal="Find the key points a technical blog post on a topic must cover.",
        backstory="You are a research analyst who distils a topic into accurate, "
                  "specific key points and never invents facts.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    outliner = Agent(
        role="Outline Writer",
        goal="Turn research notes into a clear, logical blog post outline.",
        backstory="You are an editor who structures technical posts so that each "
                  "section builds on the previous one.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    full_writer = Agent(
        role="Full Post Writer",
        goal="Expand an outline into a complete, engaging technical blog post.",
        backstory="You are a developer-advocate writer who uses active voice and "
                  "concrete examples, and expands every outline section fully.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    research_task = Task(
        description=f"Research the topic '{topic}'. Identify the 6-8 key points a "
                    "practitioner needs: what each approach is, strengths, weaknesses, "
                    "cost and maintenance trade-offs, and when to combine them.",
        expected_output="A bullet list of key points, each with one sentence of supporting detail.",
        agent=researcher,
    )
    outline_task = Task(
        description=f"Using the research notes, create a structured outline for a blog post "
                    f"titled '{topic}'. Include an introduction, 4-6 body sections with "
                    "bullet sub-points, and a conclusion.",
        expected_output="A Markdown outline with H2 section headings and bullet sub-points.",
        agent=outliner,
        context=[research_task],
    )
    writing_task = Task(
        description="Write the full blog post by expanding EVERY section of the outline into "
                    "well-developed paragraphs. Keep the outline's headings and order. "
                    "Target 700-900 words.",
        expected_output="The complete blog post in Markdown, with a title and all sections written out.",
        agent=full_writer,
        context=[outline_task],
    )

    return Crew(
        agents=[researcher, outliner, full_writer],
        tasks=[research_task, outline_task, writing_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )


def run_exercise_1() -> None:
    crew = build_blog_crew(get_llm())
    result = crew.kickoff()
    print(f"=== FINAL POST: {BLOG_TOPIC} ===")
    print(result)


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Create a code review crew: Developer (writes code), Security Reviewer (checks
# for vulnerabilities), Performance Reviewer (checks for inefficiency), and Lead
# (merges feedback into a final report). Test with a provided Python snippet.

SAMPLE_SNIPPET = '''\
import hashlib
import sqlite3


def find_user(db_path, username, password):
    conn = sqlite3.connect(db_path)
    digest = hashlib.md5(password.encode()).hexdigest()
    query = "SELECT * FROM users WHERE name = '" + username + "' AND pw = '" + digest + "'"
    return conn.execute(query).fetchall()


def unique_emails(users):
    result = []
    for u in users:
        if u["email"] not in result:
            result.append(u["email"])
    return result


def build_report(users):
    report = ""
    for u in users:
        for other in users:
            if u["email"] == other["email"]:
                report += u["name"] + ","
    return report
'''


def build_code_review_crew(llm: Any, snippet: str = SAMPLE_SNIPPET) -> Crew:
    """Developer -> (Security Reviewer, Performance Reviewer) -> Lead."""
    developer = Agent(
        role="Developer",
        goal="Present code changes clearly so reviewers can assess them.",
        backstory="You are a pragmatic Python developer who submits code for review "
                  "with a short description of what it does and what it assumes.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    security_reviewer = Agent(
        role="Security Reviewer",
        goal="Find vulnerabilities in submitted code and explain how to fix them.",
        backstory="You are an application security engineer who knows the OWASP Top 10 "
                  "and reviews code for injection, weak cryptography and resource leaks.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    performance_reviewer = Agent(
        role="Performance Reviewer",
        goal="Find inefficiencies in submitted code and propose faster alternatives.",
        backstory="You are a performance engineer who spots poor algorithmic complexity, "
                  "wasteful string handling and unnecessary repeated work.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    lead = Agent(
        role="Lead Reviewer",
        goal="Merge reviewer feedback into one prioritised, actionable report.",
        backstory="You are a tech lead who deduplicates feedback, ranks issues by "
                  "severity and gives a clear approve / request-changes decision.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    submit_task = Task(
        description="Prepare the following Python snippet for review: state its purpose in "
                    "two sentences, list the inputs it assumes, and reproduce the code "
                    f"UNCHANGED in a fenced code block.\n\n```python\n{snippet}```",
        expected_output="A review request: purpose, assumptions, and the unchanged code in a fenced block.",
        agent=developer,
    )
    security_task = Task(
        description="Review the submitted code for security vulnerabilities. For each issue "
                    "give: location (function), severity (High/Medium/Low), why it is "
                    "exploitable, and a concrete fix.",
        expected_output="A security findings list with location, severity, explanation and fix for each issue.",
        agent=security_reviewer,
        context=[submit_task],
    )
    performance_task = Task(
        description="Review the submitted code for performance problems. For each issue give: "
                    "location (function), the current complexity, why it is slow, and an "
                    "improved version.",
        expected_output="A performance findings list with location, complexity, explanation and improved code.",
        agent=performance_reviewer,
        context=[submit_task],
    )
    lead_task = Task(
        description="Merge the security and performance reviews into one final report. "
                    "Remove duplicates, order issues by severity, and finish with an overall "
                    "decision: Approve or Request Changes.",
        expected_output="A Markdown report: summary, prioritised issue list with fixes, and a final decision.",
        agent=lead,
        context=[security_task, performance_task],
    )

    return Crew(
        agents=[developer, security_reviewer, performance_reviewer, lead],
        tasks=[submit_task, security_task, performance_task, lead_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )


def run_exercise_2() -> None:
    crew = build_code_review_crew(get_llm())
    result = crew.kickoff()
    print("=== FINAL CODE REVIEW REPORT ===")
    print(result)


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Implement a custom WebSearchTool CrewAI tool that wraps a mock search function
# returning predefined results. Wire it to a Researcher agent and verify it uses
# the tool when asked a question requiring research.

MOCK_SEARCH_INDEX = [
    {
        "title": "Fine-tuning vs RAG: when to use which",
        "url": "https://mock.search/fine-tuning-vs-rag",
        "snippet": "RAG injects fresh external knowledge at query time and can cite its "
                   "sources; fine-tuning changes the model's weights to teach style, "
                   "format or domain behaviour. RAG is cheaper to keep up to date; "
                   "fine-tuning is better for consistent tone and structured output.",
        "keywords": {"fine-tuning", "finetuning", "rag", "retrieval", "difference", "differences", "compare"},
    },
    {
        "title": "LoRA: low-rank adaptation explained",
        "url": "https://mock.search/lora-explained",
        "snippet": "LoRA freezes the base model and trains small low-rank adapter matrices, "
                   "so adapters are only a few MB, train on modest hardware and can be "
                   "swapped without touching the base weights.",
        "keywords": {"lora", "peft", "adapter", "adapters", "fine-tuning", "cheaper", "cost"},
    },
    {
        "title": "Vector databases for retrieval",
        "url": "https://mock.search/vector-databases",
        "snippet": "Vector databases store embeddings and serve approximate nearest-neighbour "
                   "search. Common choices are Chroma (local), Pinecone (managed) and "
                   "Weaviate (hybrid search).",
        "keywords": {"vector", "database", "databases", "chroma", "pinecone", "weaviate", "embeddings", "rag"},
    },
    {
        "title": "Long context windows vs retrieval",
        "url": "https://mock.search/long-context-vs-retrieval",
        "snippet": "Context windows of 128k to 1M tokens let more documents fit in one prompt, "
                   "but cost and latency grow with prompt length, so retrieval is still "
                   "useful for selecting only the relevant passages.",
        "keywords": {"context", "window", "windows", "long", "tokens", "retrieval", "rag"},
    },
    {
        "title": "CrewAI processes: sequential vs hierarchical",
        "url": "https://mock.search/crewai-processes",
        "snippet": "In a sequential process tasks run in order and pass context forward; in a "
                   "hierarchical process a manager agent plans, delegates and reviews, at "
                   "the cost of extra LLM calls.",
        "keywords": {"crewai", "crew", "agents", "sequential", "hierarchical", "process", "manager"},
    },
]

# Every query the tool receives is recorded here, so we can verify the agent
# actually called it (and not just answered from its own knowledge).
SEARCH_LOG: list[str] = []


def mock_web_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Predefined-results search: ranks entries by how many query words match their
    keywords or title. Returns [] when nothing matches. No network access."""
    words = set(re.findall(r"[a-z0-9\-]+", query.lower()))
    words |= {w for word in words for w in word.split("-")}  # 'fine-tuning' also matches 'tuning'

    scored: list[tuple[int, dict]] = []
    for entry in MOCK_SEARCH_INDEX:
        entry_words = set(entry["keywords"]) | set(re.findall(r"[a-z0-9\-]+", entry["title"].lower()))
        entry_words |= {w for word in entry_words for w in word.split("-")}
        score = len(words & entry_words)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"title": e["title"], "url": e["url"], "snippet": e["snippet"]}
        for _, e in scored[:max_results]
    ]


class WebSearchInput(BaseModel):
    """Arguments the agent must supply when calling the Web Search tool."""
    query: str = Field(description="Search query, e.g. 'fine-tuning vs RAG differences'")
    max_results: int = Field(default=3, ge=1, le=5, description="Maximum number of results (1-5)")


class WebSearchTool(BaseTool):
    """Web search backed by predefined mock results (deterministic, offline)."""
    name: str = "Web Search"
    description: str = (
        "Search the web for information on AI topics. Returns a JSON list of results "
        "with title, url and snippet. Use it whenever a question needs research."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int = 3) -> str:
        SEARCH_LOG.append(query)
        results = mock_web_search(query, max_results)
        if not results:
            return json.dumps({"query": query, "results": [], "message": "No results found."})
        return json.dumps({"query": query, "results": results})


web_search_tool = WebSearchTool()

RESEARCH_QUESTION = (
    "What are the main differences between fine-tuning and RAG, and why is LoRA "
    "a cheaper way to fine-tune?"
)


def build_research_crew(llm: Any, question: str = RESEARCH_QUESTION) -> Crew:
    """A single Researcher agent that is expected to use the Web Search tool."""
    researcher = Agent(
        role="AI Researcher",
        goal="Answer research questions using the Web Search tool, never from memory alone.",
        backstory="You are a careful researcher. You ALWAYS call the Web Search tool before "
                  "answering and you cite the URL of every source you used.",
        llm=llm,
        tools=[web_search_tool],
        verbose=False,
        max_iter=5,
    )
    task = Task(
        description=f"Answer this question using ONLY information returned by the Web Search "
                    f"tool: {question}\nCall the tool at least once and list the URLs you used.",
        expected_output="A concise answer (under 200 words) followed by a 'Sources:' list of URLs.",
        agent=researcher,
    )
    return Crew(agents=[researcher], tasks=[task], process=Process.sequential,
                verbose=False, memory=False)


def check_web_search_tool_offline() -> bool:
    """Exercise the tool through _run() and its Pydantic schema - no LLM involved."""
    checks: list[tuple[str, bool]] = []

    hit = json.loads(web_search_tool._run(query="fine-tuning vs RAG differences"))
    checks.append(("relevant query returns results", len(hit["results"]) >= 1))
    checks.append(("best match is the fine-tuning vs RAG entry",
                   hit["results"][0]["url"] == "https://mock.search/fine-tuning-vs-rag"))

    limited = json.loads(web_search_tool._run(query="rag", max_results=1))
    checks.append(("max_results is respected", len(limited["results"]) == 1))

    miss = json.loads(web_search_tool._run(query="quantum gardening"))
    checks.append(("unrelated query returns no results", miss["results"] == []))

    checks.append(("every call is logged in SEARCH_LOG", len(SEARCH_LOG) == 3))

    try:
        WebSearchInput(query="x", max_results=0)
        rejected = False
    except ValidationError:
        rejected = True
    checks.append(("schema rejects max_results=0", rejected))

    SEARCH_LOG.clear()
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return all(ok for _, ok in checks)


def run_exercise_3() -> None:
    SEARCH_LOG.clear()
    crew = build_research_crew(get_llm())
    result = crew.kickoff()
    answer = str(result)

    print("=== RESEARCHER ANSWER ===")
    print(answer)

    cited = [e["url"] for e in MOCK_SEARCH_INDEX if e["url"] in answer]
    print("\n=== TOOL USAGE CHECK ===")
    print(f"tool called {len(SEARCH_LOG)} time(s): {SEARCH_LOG}")
    print(f"mock sources cited in the answer: {cited}")
    if SEARCH_LOG and cited:
        print("PASS: the agent used the Web Search tool and cited its results.")
    elif SEARCH_LOG:
        print("PARTIAL: the tool was called but no mock source URL appears in the answer.")
    else:
        print("FAIL: the agent answered without calling the Web Search tool.")


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Compare Process.sequential vs Process.hierarchical on the same 3-task
# pipeline. Measure total execution time and compare the quality of outputs.
# Document when each process type is preferable.

COMPARE_TOPIC = "Vector Databases for RAG in Production"


def _make_pipeline_tasks(topic: str, researcher: Agent | None = None, writer: Agent | None = None,
                         reviewer: Agent | None = None) -> list[Task]:
    """The same research -> write -> review pipeline for both processes. In the
    sequential run each task is pinned to an agent; in the hierarchical run the
    agents are left None so the manager decides who does what."""
    research = Task(
        description=f"Research '{topic}': what they are, how to choose one, and three "
                    "implementation best practices.",
        expected_output="Research notes in Markdown with headers and bullet points.",
        agent=researcher,
    )
    write = Task(
        description=f"Using the research, write a technical brief on '{topic}' (300-400 words).",
        expected_output="A technical brief in Markdown, 300-400 words, with headers.",
        agent=writer,
        context=[research],
    )
    review = Task(
        description="Review the brief for accuracy, completeness and clarity. Finish with a "
                    "verdict line: 'Verdict: Ready' or 'Verdict: Needs Revision'.",
        expected_output="A short review with specific feedback and a final verdict line.",
        agent=reviewer,
        context=[write],
    )
    return [research, write, review]


def _make_pipeline_agents(llm: Any) -> tuple[Agent, Agent, Agent]:
    researcher = Agent(role="AI Research Analyst", goal="Research AI topics thoroughly.",
                       backstory="Senior researcher specialising in LLM systems and RAG.",
                       llm=llm, verbose=False, max_iter=5)
    writer = Agent(role="Technical Writer", goal="Write developer-focused technical content.",
                   backstory="Technical writer with Python and ML experience.",
                   llm=llm, verbose=False, max_iter=5)
    reviewer = Agent(role="Content Quality Reviewer", goal="Ensure content is accurate and clear.",
                     backstory="Meticulous editor who reviews technical AI content.",
                     llm=llm, verbose=False, max_iter=5)
    return researcher, writer, reviewer


def build_sequential_crew(llm: Any, topic: str = COMPARE_TOPIC) -> Crew:
    researcher, writer, reviewer = _make_pipeline_agents(llm)
    return Crew(
        agents=[researcher, writer, reviewer],
        tasks=_make_pipeline_tasks(topic, researcher, writer, reviewer),
        process=Process.sequential,
        verbose=False,
        memory=False,
    )


def build_hierarchical_crew(llm: Any, topic: str = COMPARE_TOPIC) -> Crew:
    manager = Agent(
        role="AI Content Production Manager",
        goal="Coordinate a team to produce high-quality AI content efficiently.",
        backstory="Experienced content production manager who assigns work to the right "
                  "specialist and checks quality before accepting it.",
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )
    return Crew(
        agents=list(_make_pipeline_agents(llm)),  # manager is NOT in this list
        tasks=_make_pipeline_tasks(topic),
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=False,
        memory=False,
    )


@dataclass
class RunMetrics:
    """What we record for one crew run."""
    process: str
    seconds: float
    tasks_configured: int
    tasks_completed: int
    llm_requests: int | None
    total_tokens: int | None
    words: int
    headings: int
    has_verdict: bool
    output: str


def run_and_measure(label: str, crew: Crew) -> RunMetrics:
    start = time.perf_counter()
    result = crew.kickoff()
    seconds = time.perf_counter() - start

    text = str(result)
    usage = getattr(result, "token_usage", None)  # not exposed by every CrewAI version
    return RunMetrics(
        process=label,
        seconds=seconds,
        tasks_configured=len(crew.tasks),
        tasks_completed=len(getattr(result, "tasks_output", []) or []),
        llm_requests=getattr(usage, "successful_requests", None),
        total_tokens=getattr(usage, "total_tokens", None),
        words=len(text.split()),
        headings=len(re.findall(r"^#{1,6}\s", text, flags=re.MULTILINE)),
        has_verdict="verdict" in text.lower(),
        output=text,
    )


def format_comparison(runs: list[RunMetrics]) -> str:
    """Side-by-side table of the recorded metrics, one column per process."""
    def cell(value: object) -> str:
        return "n/a" if value is None else str(value)

    rows = [
        ("total time (s)", lambda r: f"{r.seconds:.1f}"),
        ("tasks configured", lambda r: r.tasks_configured),
        ("tasks completed", lambda r: r.tasks_completed),
        ("LLM requests", lambda r: r.llm_requests),
        ("total tokens", lambda r: r.total_tokens),
        ("output words", lambda r: r.words),
        ("markdown headings", lambda r: r.headings),
        ("review verdict present", lambda r: "yes" if r.has_verdict else "no"),
    ]
    width = max(len(name) for name, _ in rows) + 2
    lines = ["metric".ljust(width) + "".join(r.process.rjust(14) for r in runs)]
    lines.append("-" * len(lines[0]))
    for name, getter in rows:
        lines.append(name.ljust(width) + "".join(cell(getter(r)).rjust(14) for r in runs))
    return "\n".join(lines)


PROCESS_GUIDE = """\
When to use which process
-------------------------
Process.sequential
  + Predictable: fixed order, fixed agent per task, easy to debug and to cost out.
  + Fewer LLM calls, so lower latency and spend.
  + Best when the pipeline is known in advance (research -> write -> review),
    each step has an obvious owner, and outputs flow forward via context=[...].
  - No re-planning: if a step goes wrong, the next steps just consume it.

Process.hierarchical
  + A manager plans, assigns and validates, so it copes better with open-ended
    goals, vague task ownership, or work that may need re-delegation / retries.
  + Useful when the right specialist depends on the content, or when a quality
    gate on every result matters more than speed.
  - Extra manager calls make it slower and more expensive, and its decisions are
    less deterministic. A weak manager LLM can mis-delegate.

Rule of thumb: start sequential; move to hierarchical only when a fixed order
stops fitting the problem and the extra cost buys visible quality.
"""


def run_exercise_4() -> None:
    llm = get_llm()
    print("Running the sequential crew ...")
    sequential = run_and_measure("sequential", build_sequential_crew(llm))
    print("Running the hierarchical crew (this is the expensive one) ...")
    hierarchical = run_and_measure("hierarchical", build_hierarchical_crew(llm))

    print("\n=== COMPARISON ===")
    print(format_comparison([sequential, hierarchical]))

    for run in (sequential, hierarchical):
        print(f"\n--- {run.process} output (first 500 chars) ---")
        print(run.output[:500])

    print("\nQuality note: word count, headings and the review verdict are only rough "
          "proxies - read both outputs to judge accuracy and depth yourself.\n")
    print(PROCESS_GUIDE)


# ── Runner ───────────────────────────────────────────────────────────────────

def run_offline_checks() -> None:
    print("=== Offline checks (no API calls) ===")
    print("Exercise 3 - WebSearchTool via _run():")
    if not check_web_search_tool_offline():
        print("  one or more tool checks FAILED")

    print("\nBuilding every crew with a placeholder LLM:")
    placeholder = get_llm(offline_ok=True)
    crews = {
        "ex1 blog pipeline": build_blog_crew(placeholder),
        "ex2 code review": build_code_review_crew(placeholder),
        "ex3 research + tool": build_research_crew(placeholder),
        "ex4 sequential": build_sequential_crew(placeholder),
        "ex4 hierarchical": build_hierarchical_crew(placeholder),
    }
    for label, crew in crews.items():
        print(f"  {label}: {len(crew.agents)} agents, {len(crew.tasks)} tasks, process={crew.process}")


EXERCISES = {"1": run_exercise_1, "2": run_exercise_2, "3": run_exercise_3, "4": run_exercise_4}


if __name__ == "__main__":
    selected = sys.argv[1:]
    if "all" in selected:
        selected = list(EXERCISES)

    run_offline_checks()

    unknown = [s for s in selected if s not in EXERCISES]
    if unknown:
        print(f"\nUnknown exercise(s): {', '.join(unknown)}. Use 1, 2, 3, 4 or all.")
    elif not selected:
        print("\nPass exercise numbers (or 'all') to run them against the LLM, e.g.: "
              "python 07_exercises.py 1 3")
    elif not has_llm_access():
        print(f"\n{NO_KEY_MESSAGE}")
    else:
        print(f"\nLLM: {describe_llm()}")
        for number in selected:
            print(f"\n{'=' * 60}\nExercise {number}\n{'=' * 60}")
            EXERCISES[number]()
