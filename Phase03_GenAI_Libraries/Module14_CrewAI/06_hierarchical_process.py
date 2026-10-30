# Module 14 - CrewAI
# 14.6 Hierarchical Process with a Manager
# In a hierarchical process, a manager agent orchestrates which worker agent
# handles each task: it breaks the work down, delegates to the right specialist
# and checks the result before the crew finishes.
#
# NOTE: crew.kickoff() makes REAL, billed API calls - and the hierarchical
# process is the most expensive one, because the manager adds its own planning,
# delegation and review calls on top of the workers' calls. It requires
# OPENAI_API_KEY (or OPENROUTER_API_KEY / ANTHROPIC_API_KEY)
# in a .env file in this folder. Without a key this file only builds the crew
# and prints its structure.
# NOTE: The book lists the manager in agents=[manager, researcher, writer] AND
# passes manager_agent=manager. CrewAI rejects a manager that also appears in the
# agents list, so only the workers are listed in agents here.

from typing import Any

from _openrouter_client import NO_KEY_MESSAGE, describe_llm, get_llm, has_llm_access, require_crewai

require_crewai()

from crewai import Agent, Crew, Process, Task  # noqa: E402


def build_hierarchical_crew(llm: Any) -> Crew:
    """A manager coordinating a researcher and a writer on one technical brief."""
    # Manager - coordinates other agents
    manager = Agent(
        role="AI Content Production Manager",
        goal="Coordinate a team to produce high-quality AI content efficiently.",
        backstory="""You are an experienced content production manager who excels at
        breaking down projects, assigning work to the right specialist, and ensuring
        quality standards are met before publication.""",
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )

    # Workers
    researcher = Agent(
        role="AI Research Analyst",
        goal="Research AI topics thoroughly.",
        backstory="Senior researcher specialising in LLM systems and RAG.",
        llm=llm,
        verbose=False,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Write developer-focused technical content.",
        backstory="Technical writer with Python and ML experience.",
        llm=llm,
        verbose=False,
    )

    main_task = Task(
        description="""Produce a complete technical brief on 'Vector Databases for RAG in Production'.
        Include: what they are, how to choose one, and three implementation best practices.""",
        expected_output="A structured technical brief in Markdown, 400-500 words.",
        agent=manager,  # manager is responsible but may delegate
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[main_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=False,
    )


def print_crew_structure(crew: Crew) -> None:
    print(f"process: {crew.process}")
    print(f"manager: {crew.manager_agent.role}")
    for agent in crew.agents:
        print(f"- worker: {agent.role}")
    for i, task in enumerate(crew.tasks, start=1):
        print(f"- task {i}: {task.description.strip().splitlines()[0]}")


if __name__ == "__main__":
    if has_llm_access():
        print(f"LLM: {describe_llm()}\n")
        crew = build_hierarchical_crew(get_llm())
        print_crew_structure(crew)
        print()
        result = crew.kickoff()
        print(result)
    else:
        print(NO_KEY_MESSAGE)
        print("Showing the crew structure only (no kickoff).\n")
        print_crew_structure(build_hierarchical_crew(get_llm(offline_ok=True)))
