"""QLoRA hyperparameters for Qwen2.5-7B fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QLoRAConfig:
    """Complete QLoRA training configuration.

    Attributes:
        model_name: HuggingFace model ID.
        lora_r: LoRA rank (higher = more parameters, more expressive).
        lora_alpha: LoRA scaling factor (typically 2x lora_r).
        lora_dropout: Dropout on LoRA layers.
        target_modules: Which weight matrices to apply LoRA to.
        load_in_4bit: Enable 4-bit quantization (QLoRA).
        bnb_4bit_compute_dtype: Compute dtype for 4-bit layers.
        max_seq_length: Maximum token sequence length.
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate before backward.
        learning_rate: Peak learning rate for AdamW.
        num_train_epochs: Number of full passes over the dataset.
        warmup_ratio: Fraction of steps for linear LR warmup.
        output_dir: Directory to save checkpoints.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Run validation every N steps.
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    output_dir: str = "outputs/qlora-diagnostic"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
