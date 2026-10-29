# Module 14 - CrewAI
# 14.1 What CrewAI Is
# CrewAI is a framework for orchestrating multiple AI agents that collaborate on
# a shared goal. Each agent has a role, backstory, and set of tools. Tasks are
# assigned to agents, and a Crew coordinates execution.
#
# The key mental model:
#   Agent - a specialist (researcher, writer, reviewer) with a defined expertise and tools
#   Task  - a concrete piece of work with an expected output
#   Crew  - a team of agents that executes tasks, either sequentially or hierarchically
#
# When to use CrewAI: content pipelines, multi-step research workflows, code
# review loops, and any process where distinct specialised roles add clarity
# and quality.
#
# This file is a plain-Python tour of that mental model - it needs no API key
# and does not require crewai to be installed (it only reports whether it is).

from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class Building:
    """One CrewAI building block and what it is responsible for."""
    name: str
    is_a: str
    key_fields: tuple[str, ...]
    example: str


BUILDING_BLOCKS = [
    Building(
        name="Agent",
        is_a="a specialist with a defined expertise and tools",
        key_fields=("role", "goal", "backstory", "llm", "tools", "allow_delegation", "max_iter"),
        example="AI Research Analyst, Technical Content Writer, Content Quality Reviewer",
    ),
    Building(
        name="Task",
        is_a="a concrete piece of work with an expected output",
        key_fields=("description", "expected_output", "agent", "context"),
        example="Research RAG in 2024-2025 -> write a developer guide -> review the guide",
    ),
    Building(
        name="Crew",
        is_a="a team of agents that executes tasks",
        key_fields=("agents", "tasks", "process", "manager_agent", "memory", "verbose"),
        example="Crew(agents=[...], tasks=[...], process=Process.sequential)",
    ),
]

PROCESSES = {
    "Process.sequential": "tasks run one after another in the order listed; "
                          "context=[previous_task] feeds an earlier output forward",
    "Process.hierarchical": "a manager agent decides which worker handles each task "
                            "and can delegate / review before accepting results",
}

WHEN_TO_USE = [
    "Content pipelines (research -> write -> review)",
    "Multi-step research workflows",
    "Code review loops",
    "Any process where distinct specialised roles add clarity and quality",
]


def installed_crewai_version() -> str | None:
    """Version of the installed crewai package, or None when it is not installed."""
    try:
        return metadata.version("crewai")
    except metadata.PackageNotFoundError:
        return None


def print_mental_model() -> None:
    print("=== CrewAI mental model ===")
    for block in BUILDING_BLOCKS:
        print(f"\n{block.name} - {block.is_a}")
        print(f"  key fields: {', '.join(block.key_fields)}")
        print(f"  example:    {block.example}")

    print("\n=== Two ways a Crew can run its tasks ===")
    for name, description in PROCESSES.items():
        print(f"  {name}: {description}")

    print("\n=== When to use CrewAI ===")
    for item in WHEN_TO_USE:
        print(f"  - {item}")


if __name__ == "__main__":
    print_mental_model()
    version = installed_crewai_version()
    print(f"\ncrewai installed: {version}" if version
          else "\ncrewai is not installed yet - run: pip install crewai langchain-openai python-dotenv")
