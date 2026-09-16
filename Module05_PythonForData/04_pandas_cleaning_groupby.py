# Module 05 - Python for Data
# 5.2 Pandas - Cleaning and Transforming / GroupBy and Aggregation

import pandas as pd
import numpy as np

# ── Cleaning and Transforming ─────────────────────────────────────────────────

# Simulate a messy evaluation dataset
raw = pd.DataFrame({
    "prompt":     ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "response":   ["OK", None, "Good", "Bad", "OK"],
    "score":      [0.9, None, 0.85, 0.3, 0.88],
    "latency_ms": [410, 520, None, 390, 480],
})

# Inspect missing data
print(raw.isnull().sum())

# Fill missing values
raw["score"]      = raw["score"].fillna(raw["score"].mean())
raw["latency_ms"] = raw["latency_ms"].fillna(raw["latency_ms"].median())

# Drop rows still missing critical columns
clean = raw.dropna(subset=["response"]).copy()

# Add computed column
clean["pass"] = clean["score"] >= 0.7

print(clean)
print(f"Pass rate: {clean['pass'].mean():.0%}")


# ── GroupBy and Aggregation ────────────────────────────────────────────────────

evals = pd.DataFrame({
    "model":      ["claude", "gpt-4o", "claude", "gpt-4o", "claude", "gpt-4o"],
    "task":       ["qa", "qa", "summarise", "summarise", "code", "code"],
    "score":      [0.91, 0.88, 0.85, 0.82, 0.93, 0.90],
    "latency_ms": [420, 380, 610, 550, 340, 300],
})

# Average score per model
print(evals.groupby("model")["score"].mean())

# Multiple aggregations
summary = evals.groupby("model").agg(
    avg_score   = ("score", "mean"),
    avg_latency = ("latency_ms", "mean"),
    num_tasks   = ("task", "count"),
)
print(summary)

# Pivot table - model vs task
pivot = evals.pivot_table(
    values="score", index="model", columns="task", aggfunc="mean"
)
print(pivot)
