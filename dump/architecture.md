# Architecture — Resume Intelligence LLM System
### Design built around Qwen3-1.7B + QLoRA

---

## 1. High-level flow

```
                 ┌───────────────────────┐
   resume.txt →  │   PRE-PROCESSING       │  (text extraction, cleanup,
   resume.pdf →  │   / normalization      │   section heuristics)
                 └───────────┬───────────┘
                             ▼
                 ┌───────────────────────┐
                 │  PROMPT ASSEMBLY       │  instruction + schema + input
                 │  (fixed system prompt) │
                 └───────────┬───────────┘
                             ▼
                 ┌───────────────────────┐
                 │  Qwen3-1.7B            │  base model + LoRA adapter
                 │  (QLoRA fine-tuned,    │  non-thinking mode
                 │   4-bit or bf16)       │
                 └───────────┬───────────┘
                             ▼
                 ┌───────────────────────┐
                 │  POST-PROCESSING       │  JSON parse/repair, schema
                 │  / VALIDATION          │  validation (pydantic/jsonschema)
                 └───────────┬───────────┘
                    valid │        │ invalid
                          ▼        ▼
                  structured    retry w/ stricter
                  JSON output   decoding params, then
                                fallback error response
```

Everything downstream of "prompt assembly" is the model; everything around it is ordinary software engineering — which is deliberate, because it's what lets Part 4 (evaluation) and Part 5 (engineering) be built and tested independently of whether fine-tuning has already run.

---

## 2. Output schema (the contract the whole system is built around)

Designed to be strict enough to validate mechanically, but forgiving of missing data (nulls, not hallucinated values):

```json
{
  "name": "string | null",
  "email": "string | null",
  "phone": ["string"],
  "location": "string | null",
  "summary": "string | null",
  "education": [
    {
      "degree": "string | null",
      "institution": "string | null",
      "start_date": "string | null",
      "end_date": "string | null"
    }
  ],
  "experience": [
    {
      "title": "string | null",
      "company": "string | null",
      "start_date": "string | null",
      "end_date": "string | null",
      "description": "string | null"
    }
  ],
  "skills": ["string"],
  "certifications": ["string"],
  "links": ["string"]
}
```

This schema is defined once, in code, as a `pydantic` model — and reused in three places: (1) dataset generation (every training example's `output` must validate against it), (2) inference post-processing (reject/repair anything that doesn't validate), (3) evaluation (field-by-field diffing needs a stable shape to compare against). Defining it three separate times would be the most common source of silent bugs in this project.

---

## 3. Component design

### 3.1 Data layer (`data/`)
- `schema.py` — the pydantic schema above, single source of truth.
- `resume_dataset.json` — the ≥50 hand-curated `{instruction, input, output}` examples.
- `generate_synthetic.py` — template + LLM-assisted generator for scaling to thousands (Part 2's scaling answer, made concrete): parameterized templates → controlled field substitution → validation against `schema.py` → deduplication (embedding similarity) → human-in-the-loop spot review of a sample → train/val/test split.
- `train.json` / `val.json` / `test.json` — fixed splits; `test.json` is never touched during training or hyperparameter tuning, and doubles as the eval/regression set in `evaluation/`.

### 3.2 Training layer (`training/`)
- `config.yaml` — model name, LoRA rank/alpha/dropout, learning rate, epochs, batch size, max sequence length, quantization settings, output dir. No magic numbers in `train.py`.
- `train.py` — loads Qwen3-1.7B in 4-bit (bitsandbytes) via QLoRA, attaches LoRA adapters (`peft`) to the attention/MLP projection layers, trains on `train.json`, checkpoints against `val.json` loss, saves the adapter (not a full model copy — a few MB, not several GB).
- Target hyperparameters as a starting point: LoRA rank 16, alpha 32, dropout 0.05, lr 2e-4, 3 epochs, effective batch size 16 (via gradient accumulation on modest hardware), max sequence length 2048 (covers input resume + schema + output with margin), `enable_thinking=False` throughout so the model is never taught to emit reasoning tokens before the JSON.

### 3.3 Evaluation layer (`evaluation/`)
- `evaluate.py` — runs a model (base or fine-tuned) over `test.json`, produces a report.
- `metrics.py` — implements:
  - **JSON validity rate** (does it parse at all)
  - **Schema validity rate** (does it parse *and* match `schema.py`)
  - **Field-level exact-match accuracy** for scalar fields (name, email, phone)
  - **Set precision/recall/F1** for list fields (skills, certifications)
  - **Structured diff score** for nested fields (education, experience) — matched by best-alignment, then field-level accuracy within each matched entry
  - **Hallucination / unsupported-field rate** — cheap heuristic first (does every extracted string appear, fuzzy-matched, in the source input?), with an optional LLM-as-judge pass for the harder "text is a supported inference vs. invented" cases
- `regression_tests.py` — a small, fixed set of hand-picked adversarial cases (missing every optional field, duplicate sections, non-English name, multiple phone numbers) that must keep passing across every retrain; run automatically after training and diffed against the previous model's stored results.
- Output: a single `report.json` + human-readable `report.md` per run, so **v1 vs v2** is just "run both against the same `test.json`, place reports side by side."

### 3.4 Inference layer (`inference/`)
- `infer.py` — CLI: `python infer.py --input resume.txt` → JSON on stdout.
- `api.py` — thin FastAPI wrapper: `POST /parse-resume {"resume_text": "..."}` → schema-validated JSON, or a structured error object on failure (never a raw traceback).
- Shared `model_loader.py` — loads base model + adapter once, cached; both CLI and API call the same underlying `parse(resume_text: str) -> dict` function, so there's exactly one inference code path to test.
- Decoding: low temperature (near-greedy, e.g. 0.1–0.2) with a JSON-biased stopping strategy; on schema-validation failure, one automatic retry with temperature 0 before surfacing an error — cheap and meaningfully reduces malformed-JSON rate without masking real model errors in evaluation (retries are logged, not silently hidden from metrics).

### 3.5 Cross-cutting concerns
- **Config management**: one `config.yaml` per environment (training vs. inference), loaded via `pydantic-settings` — no hardcoded paths or hyperparameters buried in scripts.
- **Error handling**: every boundary (file not found, model load failure, invalid JSON, schema mismatch, empty input) raises a typed exception caught at the CLI/API edge and turned into a clear message — never a bare stack trace to the end user.
- **Reproducibility**: fixed random seeds in `config.yaml`; dataset splits and adapter checkpoints are versioned by filename (`adapter_v1/`, `adapter_v2/`) so `evaluation/` can always diff two named versions.

---

## 4. Why this shape

The architecture mirrors the rubric on purpose: the schema (`data/schema.py`) is the single artifact that Parts 2, 3, and 4 all depend on, so it's built first and never duplicated. Training and inference are decoupled behind one shared `parse()` function so the API/CLI (10 marks, deliberately the smallest piece) stays thin, and effort concentrates where the rubric weights it — dataset quality and the fine-tuning experiment (50 of 100 marks) — while still leaving a real, working, evaluable end-to-end system.
