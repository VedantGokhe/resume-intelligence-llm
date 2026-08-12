"""
training/train.py
=================
Fine-tune Qwen3-1.7B using QLoRA (4-bit quantization + LoRA adapters)
on the resume parsing dataset.

Reads hyperparameters from training/config.yaml.

Usage:
    python training/train.py --config training/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.schema import SYSTEM_PROMPT


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_prompt(example: dict) -> str:
    """
    Format instruction, input, and expected JSON output into Qwen3 Chat format.
    """
    instruction = example.get("instruction", INSTRUCTION_FALLBACK)
    input_text = example.get("input", "")
    output_text = (
        example["output"]
        if isinstance(example["output"], str)
        else torch.json.dumps(example["output"], ensure_ascii=False)
        if hasattr(torch, "json")
        else __import__("json").dumps(example["output"], ensure_ascii=False)
    )

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\nResume Text:\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{output_text}<|im_end|>"
    )
    return prompt


INSTRUCTION_FALLBACK = "Extract candidate information from the resume text and return it as JSON."


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-1.7B with QLoRA for Resume Parsing.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to training config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 60)
    print(f"Starting QLoRA Fine-Tuning for: {cfg['model_name']}")
    print("=" * 60)

    # 1. Quantization configuration (4-bit NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16 if cfg.get("fp16", True) else torch.float32,
        bnb_4bit_use_double_quant=cfg.get("use_nested_quant", False),
    )

    # 2. Load Base Model
    print(f"Loading base model: {cfg['model_name']} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Prepare Model for KBit Training
    model = prepare_model_for_kbit_training(model)

    # 5. Configure LoRA Adapters
    peft_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias=cfg.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Load Datasets
    data_files = {
        "train": cfg.get("train_file", "data/train.json"),
        "validation": cfg.get("val_file", "data/val.json"),
    }
    print(f"Loading data from {data_files}...")
    raw_datasets = load_dataset("json", data_files=data_files)

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["evaluation_strategy"],
        save_strategy=cfg["save_strategy"],
        fp16=cfg.get("fp16", True),
        optim=cfg.get("optim", "adamw_8bit"),
        seed=cfg.get("seed", 42),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", "none"),
    )

    # 8. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        formatting_func=lambda ex: [format_prompt(x) for x in (ex if isinstance(ex, list) else [ex])],
        max_seq_length=cfg["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    # 9. Train
    print("Starting training...")
    trainer.train()

    # 10. Save Adapter
    output_dir = cfg["output_dir"]
    print(f"Saving fine-tuned adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete! Adapter saved successfully at: {output_dir}")


if __name__ == "__main__":
    main()
