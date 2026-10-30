# Module 01 - Python Foundations
# 1.5 Scope and Closures
# Python uses LEGB scoping: Local -> Enclosing -> Global -> Built-in. Understanding
# scope prevents the most common bugs when working with configuration objects
# and shared state in AI pipelines.

API_KEY = "sk-test-xxx"   # module-level (global)


def get_client():
    base_url = "https://api.anthropic.com"   # local
    return f"Client({base_url}, key={API_KEY[:6]}...)"


print(get_client())


# Closure - a function that remembers its enclosing scope
def make_counter(start: int = 0):
    count = [start]   # mutable container so inner function can mutate it

    def increment():
        count[0] += 1
        return count[0]

    return increment


token_counter = make_counter()
print(token_counter())   # 1
print(token_counter())   # 2
print(token_counter())   # 3
