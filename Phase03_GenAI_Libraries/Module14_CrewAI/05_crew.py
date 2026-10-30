# Module 14 - CrewAI
# 14.5 Crew - Putting It Together
# A Crew bundles agents and tasks and runs them with crew.kickoff().
# Process.sequential runs the tasks in the order listed (use Process.hierarchical
# for manager-led execution - see 06_hierarchical_process.py). Agents here get
# the tools from 03_tools_for_agents.py: the researcher looks up context windows
# and the writer looks up costs for its examples.
#
# NOTE: crew.kickoff() makes REAL, billed API calls - several LLM calls per task
# (reasoning steps, tool calls), so a 3-task crew is typically a dozen or more
# requests. It requires OPENAI_API_KEY (or OPENROUTER_API_KEY /
# ANTHROPIC_API_KEY) in a .env file in this folder. Without a key this file only
# builds the crew and prints its structure.
# NOTE: memory=False keeps cross-task memory (which needs an embeddings provider)
# off; set it to True only if an embeddings key is configured.

from typing import Any

from _openrouter_client import (
    NO_KEY_MESSAGE, describe_llm, get_llm, has_llm_access, load_sibling, require_crewai,
)

require_crewai()

from crewai import Agent, Crew, Process  # noqa: E402

_tools = load_sibling("03_tools_for_agents")
context_tool, cost_tool = _tools.context_tool, _tools.cost_tool
make_tasks = load_sibling("04_tasks").make_tasks


def build_crew(llm: Any) -> Crew:
    """Researcher (with context lookup) -> Writer (with cost calculator) -> Reviewer."""
    # Recreate agents with tools
    researcher = Agent(
        role="AI Research Analyst",
        goal="Find accurate information on AI topics.",
        backstory="Senior AI research analyst with expertise in LLM systems.",
        llm=llm,
        tools=[context_tool],  # can look up context windows
        verbose=False,
        max_iter=5,
    )

    writer = Agent(
        role="Technical Content Writer",
        goal="Write clear technical content for developers.",
        backstory="Technical writer with a software engineering background.",
        llm=llm,
        tools=[cost_tool],  # can look up costs for examples
        verbose=False,
        max_iter=5,
    )

    reviewer = Agent(
        role="Content Quality Reviewer",
        goal="Ensure content is accurate and publication-ready.",
        backstory="Meticulous editor who reviews technical AI content.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

    return Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,  # tasks run in order; use Process.hierarchical for manager-led
        verbose=False,
        memory=False,                # set True to enable cross-task memory (requires embeddings)
    )


def print_crew_structure(crew: Crew) -> None:
    print(f"process: {crew.process}")
    for agent in crew.agents:
        tool_names = [tool.name for tool in agent.tools] or ["(none)"]
        print(f"- agent {agent.role}: tools = {', '.join(tool_names)}")
    for i, task in enumerate(crew.tasks, start=1):
        print(f"- task {i}: {task.agent.role}")


def run_crew(crew: Crew) -> None:
    # Kick off the crew
    result = crew.kickoff()
    print("=== CREW OUTPUT ===")
    print(result)

    # Access individual task outputs
    print("\n=== TASK OUTPUTS ===")
    for task in crew.tasks:
        print(f"\n--- {task.agent.role} ---")
        if task.output:
            print(str(task.output)[:200], "...")


if __name__ == "__main__":
    if has_llm_access():
        print(f"LLM: {describe_llm()}\n")
        crew = build_crew(get_llm())
        print_crew_structure(crew)
        print()
        run_crew(crew)
    else:
        print(NO_KEY_MESSAGE)
        print("Showing the crew structure only (no kickoff).\n")
        print_crew_structure(build_crew(get_llm(offline_ok=True)))
