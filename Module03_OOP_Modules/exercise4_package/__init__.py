# exercise4_package/__init__.py
# Exercise 4: package structure with proper __init__.py exports.
#
# This turns `exercise4_package` into a real importable package. Consumers can
# write:
#     from exercise4_package import ConversationHistory, LLMConfig
# instead of reaching into the submodules directly.

from .memory import ConversationHistory
from .config import LLMConfig

__all__ = ["ConversationHistory", "LLMConfig"]
