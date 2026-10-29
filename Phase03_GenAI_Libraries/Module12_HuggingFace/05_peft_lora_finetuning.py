# Module 12 - Hugging Face Transformers
# 12.5 PEFT / LoRA Fine-tuning
# LoRA (Low-Rank Adaptation) fine-tunes only a small number of added parameters
# - typically 1-5% of the original model size. This makes fine-tuning feasible
# on consumer GPUs and dramatically reduces storage requirements. The steps:
# load base model, configure LoRA, prepare dataset, train, save the adapter
# only, then load the adapter again for inference.
#
# NOTE: Needs `pip install peft transformers torch datasets accelerate`. The
# first run downloads distilgpt2 (~350 MB) from the Hugging Face Hub. Training
# is REAL (tiny: 5 examples, 3 epochs) but still takes a while on CPU and far
# less on a GPU, so it only runs when you pass --train:
#     python 05_peft_lora_finetuning.py --train
# Without the flag the script just prints the LoRA settings (offline, no
# download). The adapter is saved to ./data/lora_adapter (small files, not the
# full model). Swap in a bigger model like "meta-llama/Llama-3.1-8B" only on a
# GPU (and set fp16=True); that one is gated and needs HF_TOKEN in .env.
# NOTE: The book's comment quotes "trainable params: 294,912 || all params:
# 82,206,720"; the exact numbers depend on your peft/transformers versions, so
# trust what print_trainable_parameters() prints on your machine. GPT-2 uses
# Conv1D layers, so peft may warn that it switches fan_in_fan_out to True -
# that is expected and harmless here.
# NOTE: Labels are a copy of input_ids as in the book (padding tokens are not
# masked). 06_exercises.py (Exercise 4) masks padding and trains longer.

import sys
from importlib.util import find_spec
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REQUIRED_PACKAGES = ["transformers", "torch", "peft", "datasets", "accelerate"]

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "lora_output"
ADAPTER_DIR = DATA_DIR / "lora_adapter"

MODEL_NAME = "distilgpt2"  # swap for "meta-llama/Llama-3.1-8B" on GPU

# Practical LoRA targets by model family:
#   GPT-2 family:     ["c_attn"]
#   LLaMA / Mistral:  ["q_proj", "v_proj"]
#   BERT / RoBERTa:   ["query", "value"]
TARGET_MODULES = ["c_attn"]

TRAINING_EXAMPLES = [
    "Question: What is RAG? Answer: RAG stands for Retrieval-Augmented Generation.",
    "Question: What is fine-tuning? Answer: Fine-tuning trains a pre-trained model on new data.",
    "Question: What is an embedding? Answer: An embedding is a dense vector representing text meaning.",
    "Question: What is a vector database? Answer: A vector database stores embeddings for similarity search.",
    "Question: What is LangChain? Answer: LangChain is a framework for building LLM applications.",
]


def missing_packages(packages: list[str]) -> list[str]:
    """Return the subset of `packages` that is not installed."""
    return [p for p in packages if find_spec(p) is None]


# ── 1. Load base model ────────────────────────────────────────────────────────

def load_base(model_name: str = MODEL_NAME):
    """Load the tokenizer (pad token = eos token) and the base causal LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


# ── 2. Configure LoRA ─────────────────────────────────────────────────────────

def build_lora_config():
    """LoRA hyper-parameters from the book."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                         # rank - higher = more parameters, more capacity
        lora_alpha=32,               # scaling factor (usually 2x to 4x r)
        target_modules=TARGET_MODULES,  # which layers to adapt (model-specific)
        lora_dropout=0.1,
        bias="none",
    )


def wrap_with_lora(model):
    """Attach LoRA adapters to `model` and print the trainable-parameter share."""
    from peft import get_peft_model

    peft_model = get_peft_model(model, build_lora_config())
    peft_model.print_trainable_parameters()
    return peft_model


# ── 3. Prepare dataset ────────────────────────────────────────────────────────

def prepare_dataset(tokenizer, texts: list[str] = TRAINING_EXAMPLES):
    """Tokenise the texts and use input_ids as labels (causal LM objective)."""
    from datasets import Dataset

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized.map(lambda x: {"labels": x["input_ids"]})


# ── 4. Training arguments + 5. Train ──────────────────────────────────────────

def train(peft_model, tokenized, output_dir: Path = OUTPUT_DIR):
    """Run the Trainer loop on the tokenised dataset."""
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,             # True on GPU with float16 support
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",       # disable wandb/tensorboard for this example
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()
    return trainer


# ── 6. Save adapter only (small file, not the full model) ─────────────────────

def save_adapter(peft_model, tokenizer, adapter_dir: Path = ADAPTER_DIR) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")


# ── 7. Load adapter for inference ─────────────────────────────────────────────

def generate_with_adapter(
    tokenizer,
    prompt: str = "Question: What is RAG? Answer:",
    model_name: str = MODEL_NAME,
    adapter_dir: Path = ADAPTER_DIR,
) -> str:
    """Load a fresh base model, attach the saved adapter and generate."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model_name)
    tuned = PeftModel.from_pretrained(base, str(adapter_dir))
    tuned.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = tuned.generate(
            **inputs,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def run_full_pipeline() -> None:
    tokenizer, model = load_base()
    peft_model = wrap_with_lora(model)
    tokenized = prepare_dataset(tokenizer)
    train(peft_model, tokenized)
    save_adapter(peft_model, tokenizer)
    print(generate_with_adapter(tokenizer))


if __name__ == "__main__":
    missing = missing_packages(REQUIRED_PACKAGES)
    if missing:
        print(f"Missing package(s): {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
        sys.exit(0)

    if "--train" not in sys.argv[1:]:
        print(f"Base model : {MODEL_NAME}")
        print(f"LoRA       : r=8, lora_alpha=32, target_modules={TARGET_MODULES}, dropout=0.1")
        print(f"Dataset    : {len(TRAINING_EXAMPLES)} examples, 3 epochs")
        print(f"Adapter    : {ADAPTER_DIR}")
        print("\nTraining is skipped by default (downloads a model, slow on CPU).")
        print("Run with:  python 05_peft_lora_finetuning.py --train")
        sys.exit(0)

    try:
        run_full_pipeline()
    except OSError as exc:  # model could not be downloaded / found
        print(f"Could not load '{MODEL_NAME}': {exc}")
        print("Check your internet connection (needed once to download the model).")
