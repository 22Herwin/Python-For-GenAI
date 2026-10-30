# Module 04 - File I/O & APIs
# 4.1 Working with Files - Text Files
# AI pipelines constantly read and write data: prompt templates, model outputs,
# evaluation results, embeddings. The pathlib module is the modern, preferred
# way to work with file paths.

import pathlib

# pathlib is preferred over os.path in Python 3.6+
data_dir = pathlib.Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

# Write a prompt template
template = """You are a {role}.
Answer the following question concisely.

Question: {question}"""

template_file = data_dir / "qa_prompt.txt"
template_file.write_text(template, encoding="utf-8")

# Read it back and fill placeholders
loaded = template_file.read_text(encoding="utf-8")
filled = loaded.format(role="Python tutor", question="What is a generator?")
print(filled)

# List all .txt files in a directory
for f in data_dir.glob("*.txt"):
    print(f.name, f.stat().st_size, "bytes")
