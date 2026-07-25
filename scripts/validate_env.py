"""Validate that all required environment components are available (Mac-adapted).

Unlike the original CUDA-oriented check, GPU availability here is informational
only: this project runs Phases 0-3 on Apple Silicon with no GPU/MPS requirement,
and only needs CUDA once training moves to a cloud notebook in Phase 4.
"""

from __future__ import annotations

import os
import sys
import urllib.request

from dotenv import load_dotenv

load_dotenv()


def check_ollama() -> bool:
    """Verify the local Ollama server is reachable and has the agent model."""
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    tags_url = base_url.replace("/v1", "/api/tags")
    try:
        with urllib.request.urlopen(tags_url, timeout=3) as resp:
            import json

            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            print(f"  Ollama reachable at {base_url}")
            print(f"  Models available: {models or '(none pulled yet)'}")
            agent_model = os.environ.get("AGENT_MODEL", "qwen2.5:7b")
            has_model = any(agent_model in m for m in models)
            if not has_model:
                print(f"  Agent model '{agent_model}' not found — run: ollama pull {agent_model}")
            return has_model
    except Exception as e:
        print(f"  Ollama NOT reachable at {tags_url}: {e}")
        print("  Start it with: brew services start ollama")
        return False


def check_gpu() -> bool:
    """Report GPU/accelerator availability. Informational only on Mac."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  CUDA GPU: {name} ({mem:.1f} GB)")
        elif torch.backends.mps.is_available():
            print("  Apple MPS backend available (expected on Mac — CUDA not required for Phases 0-3)")
        else:
            print("  No GPU/MPS backend detected — CPU only (fine for Phases 0-3, slow for anything larger)")
        return True
    except ImportError:
        print("  PyTorch not installed")
        return False


def check_hf_token() -> bool:
    """Verify HuggingFace token for gated/rate-limited model access."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        print(f"  HF_TOKEN: set ({token[:8]}...)")
        return True
    print("  HF_TOKEN: not set — not required until Phase 4 (model download)")
    return True


def check_disk_space() -> bool:
    """Verify sufficient disk space for model weights."""
    import shutil

    total, used, free = shutil.disk_usage("/")
    free_gb = free / 1e9
    print(f"  Disk free: {free_gb:.1f} GB (need >= 50 GB for 7B model + data)")
    return free_gb >= 50


def main() -> None:
    print("Validating environment (Mac / free-cost build)...\n")
    checks = [
        ("Ollama + agent model", check_ollama),
        ("GPU / accelerator (informational)", check_gpu),
        ("HuggingFace token", check_hf_token),
        ("Disk space", check_disk_space),
    ]
    all_pass = True
    for name, check_fn in checks:
        print(f"Checking {name}:")
        result = check_fn()
        if not result:
            all_pass = False
        print()

    if all_pass:
        print("All checks passed. Ready to proceed.")
    else:
        print("Some checks failed. Fix the above issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
