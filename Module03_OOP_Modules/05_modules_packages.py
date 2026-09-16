# Module 03 - OOP & Modules
# 3.5 Module and Package Structure
# As your AI project grows, organise code into modules (single .py files) and
# packages (directories with __init__.py). A well-structured package makes
# imports clean and code reusable across projects.
#
# This file is a REFERENCE, not something meant to run standalone (the
# `my_ai_project` package it imports from does not exist in this folder). See
# `exercise4_package/` in this same directory for a small, real, runnable
# example package built from the ConversationHistory and LLMConfig classes.

RECOMMENDED_PROJECT_LAYOUT = """
my_ai_project/
    src/
        my_ai_project/
            __init__.py
            config.py              <- LLMConfig and settings
            clients/
                __init__.py
                anthropic.py       <- AnthropicClient
                openai.py          <- OpenAIClient
            retrieval/
                __init__.py
                chunker.py         <- chunk_text generator
                embedder.py        <- embed_texts function
            pipeline.py            <- main RAG pipeline
    tests/
        test_config.py
        test_chunker.py
    pyproject.toml
    README.md
"""

# ── Importing Correctly (illustrative - not runnable as-is) ───────────────────
#
# # Absolute imports - preferred
# from my_ai_project.config import LLMConfig
# from my_ai_project.clients.anthropic import AnthropicClient
#
# # Import specific names
# from my_ai_project.retrieval.chunker import chunk_text
#
# # Import module as alias
# import my_ai_project.retrieval.embedder as embedder
# vectors = embedder.embed_texts(["hello", "world"])

# NOTE:
# Use pyproject.toml (PEP 517/518) for all new projects - not setup.py. The
# standard tool chain is: hatch or poetry for packaging, ruff for linting, mypy
# for type checking, and pytest for tests.

if __name__ == "__main__":
    print(RECOMMENDED_PROJECT_LAYOUT)
