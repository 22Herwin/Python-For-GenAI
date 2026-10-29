# Module 12 - Hugging Face Transformers
# 12.4 AutoModel - Loading Models Directly
# Instead of the high-level pipeline, load the tokenizer and model yourself with
# AutoTokenizer / AutoModelForCausalLM for full control over generation.
#
# NOTE: Needs `pip install transformers torch`. The first run downloads the
# distilgpt2 weights (~350 MB) from the Hugging Face Hub, so internet is
# required once. Small enough to run on CPU (float32); on a GPU use float16.
# NOTE: The book passes `torch_dtype=`. Newer transformers releases may rename
# or deprecate this argument (`dtype=`); we keep the book's spelling and the
# call below is the only place to change if your version warns about it.

import sys
from importlib.util import find_spec

from dotenv import load_dotenv

load_dotenv()

REQUIRED_PACKAGES = ["transformers", "torch"]


def missing_packages(packages: list[str]) -> list[str]:
    """Return the subset of `packages` that is not installed."""
    return [p for p in packages if find_spec(p) is None]


def generate_locally(
    prompt: str = "The key advantage of local LLMs is",
    model_name: str = "distilgpt2",
    max_new_tokens: int = 40,
) -> str:
    """Load a small causal LM and sample a continuation of `prompt`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load a small model for local inference (CPU-friendly)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # float16 for GPU, float32 for CPU
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    missing = missing_packages(REQUIRED_PACKAGES)
    if missing:
        print(f"Missing package(s): {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
        sys.exit(0)

    try:
        print(generate_locally())
    except OSError as exc:  # model could not be downloaded / found
        print(f"Could not load 'distilgpt2': {exc}")
        print("Check your internet connection (needed once to download the model).")
