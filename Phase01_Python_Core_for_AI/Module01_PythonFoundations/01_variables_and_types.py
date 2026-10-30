# Module 01 - Python Foundations
# 1.2 Variables and Data Types
# Python is dynamically typed - you do not declare a type; Python infers it at
# runtime. Use type hints (PEP 484) for readability and IDE support, especially
# in functions.

# Core scalar types
model_name: str = "Hello, World!"
temperature: float = 0.7
max_tokens: int = 1024
is_streaming: bool = True

# Check types at runtime
print(f"Type of {model_name}: {type(model_name)}")  # <class 'str'>
print(f"Type of {temperature}: {type(temperature)}") # <class 'float'>

# None - the absence of a value
response = None
print(f"Response is None: {response is None}")      # True