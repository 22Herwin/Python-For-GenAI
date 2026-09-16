# Module 05 - Python for Data
# 5.2 Pandas
# Pandas provides the DataFrame - a labelled, 2D table built on NumPy. It is the
# standard tool for loading, cleaning, exploring, and transforming tabular data
# before feeding it to AI models.

import pandas as pd

# ── Creating and Inspecting DataFrames ────────────────────────────────────────

data = [
    {"model": "gpt-4o",            "provider": "OpenAI",    "context_k": 128,  "cost_input": 2.50},
    {"model": "claude-sonnet-4-5", "provider": "Anthropic", "context_k": 200,  "cost_input": 3.00},
    {"model": "gemini-1.5-pro",    "provider": "Google",    "context_k": 1000, "cost_input": 1.25},
    {"model": "llama-3.1-70b",     "provider": "Meta",      "context_k": 128,  "cost_input": 0.00},
]

df = pd.DataFrame(data)
print(df.shape)  # (4, 4)
print(df.dtypes)
print(df.head())
print(df.describe())


# ── Selection and Filtering ───────────────────────────────────────────────────

df2 = pd.DataFrame([
    {"model": "gpt-4o",            "context_k": 128,  "cost_input": 2.50},
    {"model": "claude-sonnet-4-5", "context_k": 200,  "cost_input": 3.00},
    {"model": "gemini-1.5-pro",    "context_k": 1000, "cost_input": 1.25},
    {"model": "llama-3.1-70b",     "context_k": 128,  "cost_input": 0.00},
])

# Select column
print(df2["model"].tolist())

# Filter rows
affordable = df2[df2["cost_input"] < 2.0]
print(affordable)

# Multiple conditions
big_and_cheap = df2[(df2["context_k"] >= 128) & (df2["cost_input"] < 2.0)]
print(big_and_cheap[["model", "context_k", "cost_input"]])

# loc (label-based) vs iloc (position-based)
print(df2.loc[0, "model"])   # 'gpt-4o'
print(df2.iloc[0, 0])        # 'gpt-4o'
