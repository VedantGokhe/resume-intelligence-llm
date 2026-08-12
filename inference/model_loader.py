"""
inference/model_loader.py
=========================
Shared model loading and parsing logic for CLI and API.

Loads base model Qwen3-1.7B and attached LoRA adapter once (cached).
Applies temperature=0.1 decoding and Pydantic schema validation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import SYSTEM_PROMPT, ResumeSchema, parse_resume_output

_MODEL_CACHE = None
_TOKENIZER_CACHE = None


def load_model_and_tokenizer(
    base_model_name: str = "Qwen/Qwen3-1.7B",
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
):
    """
    Loads base model + optional LoRA adapter.
    """
    global _MODEL_CACHE, _TOKENIZER_CACHE

    if _MODEL_CACHE is not None and _TOKENIZER_CACHE is not None:
        return _MODEL_CACHE, _TOKENIZER_CACHE

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer for {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {base_model_name}...")
    if load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # CPU or standard loading fallback
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
        )

    # Attach LoRA adapter if provided
    if adapter_path and os.path.exists(adapter_path):
        print(f"Attaching LoRA adapter from {adapter_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    _MODEL_CACHE = model
    _TOKENIZER_CACHE = tokenizer
    return model, tokenizer


def parse_resume(
    resume_text: str,
    adapter_path: Optional[str] = "./adapter_v1",
    base_model_name: str = "Qwen/Qwen3-1.7B",
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Main parsing function. Takes raw text -> returns structured dict matching ResumeSchema.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    # Try loading model
    try:
        model, tokenizer = load_model_and_tokenizer(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
        )

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nExtract candidate information from the resume text:\n\n{resume_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Clean JSON markdown fences if present
        cleaned = raw_output
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        validated = parse_resume_output(data)
        return json.loads(validated.model_dump_json())

    except Exception as e:
        # Structured error or fallback handling
        print(f"[Warning] Model inference failed or model not found ({e}). Returning structured schema error.")
        return {
            "error": "Inference error or invalid JSON generated",
            "details": str(e),
            "fallback_schema": json.loads(ResumeSchema().model_dump_json())
        }
