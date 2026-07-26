"""Merge a LoRA adapter into the base model for deployment.

Runs on CPU (no bitsandbytes/GPU needed for the merge itself, though it does
need enough RAM/disk to hold the full-precision base model -- ~15GB for
Qwen2.5-7B in bf16). Typically run right after train_sft.py or train_dpo.py
on the same Kaggle instance, before downloading the merged model.
"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_save(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """Merge LoRA adapter weights into the base model.

    Args:
        base_model_name: HuggingFace model ID for the base model.
        adapter_path: Path to the trained LoRA adapter directory.
        output_path: Path to save the merged full-precision model.
    """
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    merge_and_save(
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="outputs/qlora-diagnostic",
        output_path="outputs/merged-diagnostic-v1",
    )
