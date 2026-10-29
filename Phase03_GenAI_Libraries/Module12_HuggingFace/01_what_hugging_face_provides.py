# Module 12 - Hugging Face Transformers
# 12.1 What Hugging Face Provides
# The Hugging Face ecosystem is the central hub for open-source AI models. This
# file summarises what the `transformers` library offers, when to prefer it over
# hosted APIs, and checks which of this module's libraries are installed.
#
# NOTE: This script only inspects installed packages - it does NOT import the
# heavy libraries, download any model, or touch the network. The later files in
# this module need `transformers` and `torch` (plus `peft`, `datasets` and
# `accelerate` for 12.5) and download models from the Hugging Face Hub. An
# HF_TOKEN in a .env file is optional (see .env.example): it only matters for
# gated/private models or higher Hub rate limits.

from importlib import metadata
from importlib.util import find_spec

from dotenv import load_dotenv

load_dotenv()


# ── What the library provides ──────────────────────────────────────────────────

COMPONENTS: dict[str, str] = {
    "Pipeline API": "high-level inference for 30+ tasks in 2-3 lines of code",
    "AutoModel / AutoTokenizer": "load any model from the Hub with a single string",
    "Trainer API": "fine-tune models with minimal boilerplate",
    "PEFT": "parameter-efficient fine-tuning (LoRA, QLoRA) that trains a fraction of weights",
}

WHEN_TO_USE_OVER_HOSTED_APIS: dict[str, str] = {
    "Data privacy": "model runs locally, no data leaves your infrastructure",
    "Cost at scale": "after the inference hardware cost, no per-token fees",
    "Customisation": "fine-tune on your domain data",
    "Offline": "works without internet after model download",
}

# Import name -> pip package name (used to build the install hint below).
MODULE_12_PACKAGES: dict[str, str] = {
    "transformers": "transformers",
    "torch": "torch",
    "peft": "peft",
    "datasets": "datasets",
    "accelerate": "accelerate",
}


# ── Environment check ──────────────────────────────────────────────────────────

def installed_version(package: str) -> str | None:
    """Return the installed version of `package`, or None if it is missing.
    Uses importlib only - the package itself is never imported."""
    if find_spec(package) is None:
        return None
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def missing_packages(packages: list[str]) -> list[str]:
    """Return the subset of `packages` that is not installed."""
    return [p for p in packages if find_spec(p) is None]


def install_hint(packages: list[str]) -> str:
    """Build the `pip install ...` command for the given import names."""
    return "pip install " + " ".join(MODULE_12_PACKAGES.get(p, p) for p in packages)


def print_overview() -> None:
    print("=== What the transformers library provides ===")
    for name, description in COMPONENTS.items():
        print(f"  {name:<28s} {description}")

    print("\n=== When to use Hugging Face over hosted APIs ===")
    for name, description in WHEN_TO_USE_OVER_HOSTED_APIS.items():
        print(f"  {name:<28s} {description}")


def print_environment() -> None:
    print("\n=== Module 12 library check ===")
    for package in MODULE_12_PACKAGES:
        version = installed_version(package)
        status = f"installed ({version})" if version else "NOT installed"
        print(f"  {package:<14s} {status}")

    missing = missing_packages(list(MODULE_12_PACKAGES))
    if missing:
        print(f"\nTo run the rest of this module: {install_hint(missing)}")
    else:
        print("\nAll Module 12 libraries are installed.")


if __name__ == "__main__":
    print_overview()
    print_environment()
