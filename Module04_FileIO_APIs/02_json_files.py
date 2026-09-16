# Module 04 - File I/O & APIs
# 4.1 Working with Files - JSON

import json
import pathlib

# Save model evaluation results
results = {
    "model":         "claude-sonnet-4-5",
    "benchmark":     "MMLU",
    "scores":        {"science": 0.91, "math": 0.88, "history": 0.85},
    "total_samples": 14042,
    "timestamp":     "2025-01-15T09:30:00Z",
}

out = pathlib.Path(__file__).parent / "data" / "results.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2), encoding="utf-8")

# Read back and use
data = json.loads(out.read_text(encoding="utf-8"))
print(f"Model: {data['model']}")
avg = sum(data["scores"].values()) / len(data["scores"])
print(f"Average score: {avg:.2%}")
