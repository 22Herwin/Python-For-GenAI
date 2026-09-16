# Module 03 - OOP & Modules
# 3.6 Exercise 4 - demo of the packaged ConversationHistory and LLMConfig.
#
# Run this file from the Module03_OOP_Modules directory (or any location where
# `exercise4_package` is importable) to see the clean, package-level import
# in action.

from exercise4_package import ConversationHistory, LLMConfig

history = ConversationHistory(max_turns=5, system_prompt="Be concise.")
history.add("user", "What is a package in Python?")
history.add("assistant", "A directory with an __init__.py that groups modules.")
print(history)

cfg = LLMConfig(model="claude-sonnet-4-5", temperature=0.4)
print(cfg.as_dict)
