# Module 14 - CrewAI
# 14.2 Agents
# An Agent is defined by a role, a goal and a backstory - together they act as
# the agent's system prompt. llm selects the model, allow_delegation lets the
# agent hand work to teammates, and max_iter caps its reasoning iterations.
#
# NOTE: Constructing agents makes no API call (a placeholder LLM is used when no
# key is set). Actually running them happens in 05_crew.py via crew.kickoff(),
# which makes REAL, billed API calls and needs OPENAI_API_KEY (or one of the
# alternatives in .env.example) in a .env file in this folder.

from typing import Any

from _openrouter_client import describe_llm, get_llm, has_llm_access, require_crewai

require_crewai()

from crewai import Agent  # noqa: E402


def build_agents(llm: Any) -> tuple[Agent, Agent, Agent]:
    """Return (researcher, writer, reviewer) exactly as defined in section 14.2."""
    # Research specialist
    researcher = Agent(
        role="AI Research Analyst",
        goal="Find accurate and up-to-date information on AI topics.",
        backstory="""You are a senior AI research analyst with 10 years of experience
        reading and synthesising academic papers, technical blogs, and release notes.
        You are known for being precise, citing sources, and never making things up.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,  # this agent does not delegate to others
        max_iter=5,              # max reasoning iterations before stopping
    )

    # Writing specialist
    writer = Agent(
        role="Technical Content Writer",
        goal="Write clear, accurate, and engaging technical content for developers.",
        backstory="""You are a technical writer with a background in software engineering.
        You excel at turning complex AI concepts into practical, well-structured guides.
        You always write in active voice, use concrete examples, and avoid jargon.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # Quality reviewer
    reviewer = Agent(
        role="Content Quality Reviewer",
        goal="Ensure content is accurate, complete, and publication-ready.",
        backstory="""You are a meticulous editor who reviews technical content for
        accuracy, clarity, and completeness. You check facts, flag unsupported claims,
        and ensure the writing matches the target audience's level.""",
        llm=llm,
        verbose=True,
        allow_delegation=True,  # reviewer can delegate back to writer for revisions
    )

    return researcher, writer, reviewer


def print_agent(agent: Agent) -> None:
    first_line = agent.backstory.strip().splitlines()[0]
    print(f"- {agent.role}")
    print(f"    goal:             {agent.goal}")
    print(f"    backstory:        {first_line} ...")
    print(f"    allow_delegation: {agent.allow_delegation}")
    print(f"    max_iter:         {agent.max_iter}")


if __name__ == "__main__":
    if has_llm_access():
        print(f"LLM: {describe_llm()}\n")
    else:
        print("No API key set - building agents with a placeholder LLM (never called).\n")

    researcher, writer, reviewer = build_agents(get_llm(offline_ok=True))
    for agent in (researcher, writer, reviewer):
        print_agent(agent)
