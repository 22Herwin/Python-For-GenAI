# Module 05 - Python for Data
# 5.1 NumPy
# NumPy is the foundation of numerical computing in Python. Embedding vectors,
# similarity matrices, and attention weight arrays are all NumPy arrays under the
# hood in nearly every AI library.

import numpy as np

# ── Arrays and Dtypes ─────────────────────────────────────────────────────────

# Creating arrays
scores = np.array([0.91, 0.76, 0.88, 0.65, 0.95], dtype=np.float32)
print(scores.dtype, scores.shape)  # float32 (5,)

# Zeros, ones, ranges
zeros    = np.zeros((3, 4))                     # 3 rows, 4 cols of 0.0
rng_vals = np.arange(0, 1.0, 0.1)                # [0.0, 0.1, ..., 0.9]
linspace = np.linspace(0, 1, 5)                  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Random - use a seeded Generator for reproducibility
rng = np.random.default_rng(seed=42)
mock_embedding = rng.standard_normal(1536)  # 1536-dim like text-embedding-3-small
print(f"Embedding shape: {mock_embedding.shape}, mean: {mock_embedding.mean():.4f}")


# ── Shape, Reshape, Indexing ──────────────────────────────────────────────────

# Simulate 4 document embeddings of dimension 8
rng = np.random.default_rng(42)
embeddings = rng.standard_normal((4, 8))
print("Shape:", embeddings.shape)  # (4, 8)
print("First embedding:", embeddings[0])
print("First 3 dims of all docs:\n", embeddings[:, :3])

# Reshape
flat = embeddings.flatten()      # (32,)
back = flat.reshape(4, 8)        # (4, 8)

# Boolean indexing
similarity_scores = np.array([0.91, 0.43, 0.78, 0.55])
above_threshold   = embeddings[similarity_scores > 0.7]
print(f"Docs above 0.7 similarity: {above_threshold.shape[0]}")
