"""DPO training script -- continues from an SFT LoRA checkpoint using TRL's DPOTrainer.

Requires a CUDA GPU with bitsandbytes -- does NOT run on Apple Silicon.
Intended to run on a free Kaggle T4 notebook, AFTER train_sft.py has
produced a checkpoint at QLoRAConfig.output_dir (see
notebooks/04_finetune_kaggle.ipynb).

VERSION CAVEAT: written against trl==1.9.0 (this repo's local pip-installed
version) but NOT executed end to end -- that needs bitsandbytes + a CUDA
GPU this Mac doesn't have. DPOTrainer/DPOConfig's exact constructor
signature has changed across TRL releases (tokenizer vs. processing_class,
where max_length lives). If Kaggle's trl version raises a TypeError on
unexpected kwargs, check `pip show trl` there and adjust accordingly.

Current data reality: data/dpo/train.jsonl has exactly 1 pair (see
docs/components/04-training-data-pipeline.md) -- nowhere near enough to
meaningfully shift model behavior. Running this now mainly proves the
training pipeline works end to end; treat any resulting accuracy change as
noise until more human-reviewed corrections accumulate.
"""

from __future__ import annotations

import json

import torch
import wandb
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from src.finetune.qlora_config import QLoRAConfig


def load_dpo_dataset(path: str) -> Dataset:
    """Load a {prompt, chosen, rejected} JSONL file into a HuggingFace Dataset."""
    records = [json.loads(line) for line in open(path) if line.strip()]
    return Dataset.from_list(records)


def train(
    config: QLoRAConfig,
    sft_adapter_path: str = "outputs/qlora-diagnostic",
    dpo_train_path: str = "data/dpo/train.jsonl",
    output_dir: str = "outputs/dpo-diagnostic",
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 1,
) -> None:
    """Run DPO training on top of an existing SFT LoRA adapter.

    Args:
        config: QLoRAConfig (reused for model_name/quantization/LoRA settings).
        sft_adapter_path: Path to the LoRA adapter produced by train_sft.py.
        dpo_train_path: Path to the DPO preference-pairs JSONL.
        output_dir: Directory to save the DPO-trained adapter.
        beta: DPO temperature -- controls how strongly the model is pushed
            away from the reference (SFT) policy. Lower = more conservative.
        learning_rate: DPO typically wants a much lower LR than SFT.
        num_train_epochs: DPO overfits fast on small preference sets; keep this low.
    """
    wandb.init(
        project="logcat-intelligence-engine",
        name="dpo-diagnostic",
        config={"beta": beta, "learning_rate": learning_rate, "num_train_epochs": num_train_epochs},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Continue from the SFT adapter, not the raw base model -- DPO refines
    # what SFT already taught rather than starting over from scratch.
    model = PeftModel.from_pretrained(base_model, sft_adapter_path, is_trainable=True)

    train_dataset = load_dpo_dataset(dpo_train_path)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        beta=beta,
        max_length=config.max_seq_length,
        bf16=True,
        report_to="wandb",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    wandb.finish()


if __name__ == "__main__":
    train(QLoRAConfig())
