# Module 14 - CrewAI
# 14.4 Tasks
# A Task is a concrete piece of work: a description, an expected_output and the
# agent responsible for it. context=[other_task] wires one task's output into
# another as input - CrewAI injects the previous output into the agent's
# context automatically.
#
# NOTE: Building tasks makes no API call (a placeholder LLM is used when no key
# is set). The agents come from 02_agents.py. Running the tasks needs a Crew -
# see 05_crew.py, which makes REAL, billed API calls.

from _openrouter_client import describe_llm, get_llm, has_llm_access, load_sibling, require_crewai

require_crewai()

from crewai import Agent, Task  # noqa: E402

build_agents = load_sibling("02_agents").build_agents


def make_tasks(researcher: Agent, writer: Agent, reviewer: Agent) -> tuple[Task, Task, Task]:
    """Return (research_task, writing_task, review_task), chained with context=[...]."""
    research_task = Task(
        description="""Research the current state of RAG (Retrieval-Augmented Generation) in 2024-2025.
        Cover:
        1. What RAG is and how it works
        2. The main variants (naive RAG, advanced RAG, modular RAG)
        3. Common production challenges and how practitioners solve them
        4. Three real-world use cases with specific results

        Be specific and factual. Avoid vague generalities.""",
        expected_output="""A structured research report with:
        - A clear definition and architecture overview
        - Descriptions of the three RAG variants with key differences
        - Three specific production challenges with concrete solutions
        - Three real-world use cases with measurable outcomes
        Format: Markdown with headers.""",
        agent=researcher,
    )

    writing_task = Task(
        description="""Using the research report provided, write a developer guide titled
        'Building Production RAG Systems'.

        The guide should:
        - Be written for senior developers with Python experience
        - Include a conceptual overview, implementation considerations, and common pitfalls
        - Use active voice and concrete examples
        - Be 600-800 words""",
        expected_output="A complete, publication-ready developer guide in Markdown format.",
        agent=writer,
        context=[research_task],  # this task depends on research_task output
    )

    review_task = Task(
        description="""Review the developer guide for:
        1. Technical accuracy - flag any claims that are incorrect or unsupported
        2. Completeness - identify important topics that are missing
        3. Clarity - suggest any sections that need rewriting
        4. Audience fit - check that the level is appropriate for senior developers

        Provide a structured review with specific, actionable feedback.""",
        expected_output="""A quality review report containing:
        - Overall assessment (Ready / Needs Revision)
        - Specific accuracy issues (if any) with corrections
        - Missing content recommendations
        - Clarity improvement suggestions
        - Final verdict""",
        agent=reviewer,
        context=[writing_task],
    )

    return research_task, writing_task, review_task


def print_task_graph(tasks: list[Task]) -> None:
    """Show each task, who owns it and which earlier tasks feed it."""
    names = {id(task): f"task {i}" for i, task in enumerate(tasks, start=1)}
    for task in tasks:
        feeds_from = [names[id(t)] for t in (task.context or []) if id(t) in names]
        summary = task.description.strip().splitlines()[0]
        print(f"- {names[id(task)]} [{task.agent.role}]")
        print(f"    {summary}")
        print(f"    context from: {', '.join(feeds_from) if feeds_from else '(none - first task)'}")


if __name__ == "__main__":
    if has_llm_access():
        print(f"LLM: {describe_llm()}\n")
    else:
        print("No API key set - building tasks with a placeholder LLM (never called).\n")

    agents = build_agents(get_llm(offline_ok=True))
    tasks = make_tasks(*agents)
    print_task_graph(list(tasks))
