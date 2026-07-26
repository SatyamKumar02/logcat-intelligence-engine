"""SFT training script with QLoRA and wandb logging.

Requires a CUDA GPU with bitsandbytes (see requirements.cloud.txt) -- does
NOT run on Apple Silicon. Intended to run on a free Kaggle T4 notebook; see
notebooks/04_finetune_kaggle.ipynb.

Before spending GPU hours on this, validate the data pipeline locally with
scripts/validate_finetune_data.py (no GPU needed).

VERSION NOTE: uses trl.SFTConfig (not transformers.TrainingArguments) as the
args class for SFTTrainer, with dataset_text_field/max_length living on
SFTConfig -- verified directly against this repo's installed trl==1.9.0 via
`inspect.signature()` (SFTTrainer.__init__ no longer accepts
dataset_text_field/max_seq_length as direct kwargs in this version). The
GPU training itself has NOT been executed end to end -- that requires
bitsandbytes + a CUDA GPU, which this Mac doesn't have. If Kaggle's
preinstalled trl version differs enough to raise a TypeError, check
`pip show trl` there and re-verify SFTConfig's fields the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.finetune.qlora_config import QLoRAConfig


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load a ShareGPT-format JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_conversation(example: dict) -> dict:
    """Format a ShareGPT conversation into a single text string.

    Ignores any extra keys (e.g. the "metadata" field TraceConverter attaches
    for stats.py, see src/training/trace_recorder.py) -- only "conversations"
    matters here.
    """
    parts = []
    for turn in example.get("conversations", []):
        role = turn.get("from", "")
        content = turn.get("value", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "human":
            parts.append(f"<|user|>\n{content}")
        elif role == "gpt":
            parts.append(f"<|assistant|>\n{content}")
    return {"text": "\n".join(parts)}


def train(config: QLoRAConfig) -> None:
    """Run QLoRA SFT training.

    Args:
        config: QLoRAConfig with all hyperparameters.
    """
    wandb.init(
        project="logcat-intelligence-engine",
        name=f"qlora-sft-r{config.lora_r}",
        config=vars(config),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset_from_jsonl("data/sft/train.jsonl")
    val_dataset = load_dataset_from_jsonl("data/sft/val.jsonl")
    train_dataset = train_dataset.map(format_conversation)
    val_dataset = val_dataset.map(format_conversation)

    training_args = SFTConfig(
        output_dir=config.output_dir,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,  # deprecated in favor of warmup_steps, but still functional (verified locally)
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        fp16=False,
        bf16=True,
        report_to="wandb",
        save_total_limit=3,
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    wandb.finish()


if __name__ == "__main__":
    train(QLoRAConfig())
