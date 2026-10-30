# Module 02 - Data Structures & Comprehensions
# 2.3 Sets
# Sets are unordered collections of unique items. Use them to deduplicate retrieved
# documents, track visited URLs, or compute overlap between keyword sets.

# Deduplication
retrieved_doc_ids = ["doc_3", "doc_1", "doc_3", "doc_7", "doc_1"]
unique_ids = set(retrieved_doc_ids)
print(unique_ids)   # {'doc_1', 'doc_3', 'doc_7'}

# Set operations - useful for keyword and topic analysis
gpt4_topics = {"coding", "math", "reasoning", "vision"}
claude_topics = {"coding", "writing", "reasoning", "safety"}

both     = gpt4_topics & claude_topics   # intersection
either   = gpt4_topics | claude_topics   # union
gpt_only = gpt4_topics - claude_topics   # difference

print("Both:    ", both)
print("Either:  ", either)
print("GPT only:", gpt_only)
