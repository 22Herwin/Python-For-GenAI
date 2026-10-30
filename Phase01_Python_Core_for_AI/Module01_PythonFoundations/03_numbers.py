# Module 01 - Python Foundations
# 1.2 Variables and Data Types - Numbers

# Integer arithmetic
tokens_used = 450
tokens_limit = 1024
remaining = tokens_limit - tokens_used   # 574

# Float arithmetic
cost_per_token = 0.000003
total_cost = tokens_used * cost_per_token
print(f"Cost: ${total_cost:.6f}")

# Integer division and modulo
batches = tokens_used // 100   # 4
leftover = tokens_used % 100   # 50
print(f"Batches: {batches}, leftover: {leftover}")

# Built-in math
import math

print(math.log2(512))   # 9.0 - useful in information theory
print(math.ceil(3.1))   # 4

# NOTE:
# Use underscores in large numeric literals for readability: 1_000_000
# instead of 1000000. Python ignores the underscores.
large_number = 1_000_000
print(large_number)
