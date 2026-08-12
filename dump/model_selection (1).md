# Model Selection — Resume Intelligence LLM
### Replacing GPT-4 Mini for a domain-specific resume parsing system

---

## 1. Task Requirements (framing the decision)

Before picking a model, it's worth being explicit about what resume parsing actually demands, because that's what should drive the choice — not leaderboard rank.

| Requirement | Why it matters for resumes |
|---|---|
| Reliable instruction-following | The model must consistently map free text → a fixed schema, not "chat" |
| Strong structured/JSON output | Downstream systems (ATS, search, ranking) consume JSON directly; malformed output breaks the pipeline |
| Moderate context window | Resumes are rarely >2–3 pages (~1,500–3,000 tokens), so we don't need 1M-token context — we need *reliability* within a modest window |
| Low hallucination tendency | Inventing a phone number or a skill is worse than leaving a field null |
| Cheap to fine-tune and serve | This is a narrow, high-volume, low-margin task (parsing, not reasoning) — it doesn't justify a 70B+ model |
| Permissive license | Needs to be usable in a commercial ATS/HR product without legal ambiguity |

This is a **narrow extraction task**, not an open-ended generation or reasoning task. That reframes the whole search: we are not looking for the smartest model, we're looking for the smallest model that can be taught to be *reliable* at one job.

---

## 2. Selected Model: **Qwen3‑1.7B (Instruct/base, dense)**

**Exact checkpoint:** `Qwen/Qwen3-1.7B` (Hugging Face), Apache 2.0 license.

### Why this model

- **Parameter size: 1.7B** — squarely inside the assessment's preferred 0.5B–3B fine-tuning range. Large enough to reliably follow a schema-following instruction and handle varied resume layouts; small enough to fine-tune and serve cheaply.
- **Context window: 32,768 tokens** natively. A resume is typically 500–1,500 tokens once converted to text — so even multi-page resumes, cover letters, or few-shot examples in the prompt fit comfortably, with huge headroom left for the instruction + schema + output.
- **Instruction-following & structured output are first-class**, not an afterthought. Qwen3's dense line was trained with explicit tool-use, JSON-schema-following, and agentic-output objectives, and this carries down to the small dense checkpoints (0.6B/1.7B/4B) via strong-to-weak distillation from the larger Qwen3 models — this is exactly the skill resume parsing needs.
- **Dual-mode (thinking / non-thinking)** — for a deterministic extraction task we want *non-thinking* mode: fast, low-token, no chain-of-thought bloat in the output. The same checkpoint supports switching this off via `enable_thinking=False`, so we don't pay for reasoning tokens we don't need.
- **License: Apache 2.0** — fully permissive for commercial use, fine-tuning, redistribution of fine-tuned weights, and internal deployment inside an HR product with no attribution or copyleft obligations. This matters because a resume platform is a commercial product, not a research demo.
- **Ecosystem maturity** — native `transformers`/PEFT support, first-class QLoRA/LoRA recipes via Unsloth, TRL, and Axolotl, GGUF/AWQ quantized builds for CPU/edge inference, and broad community documentation. For a project where *fine-tuning it correctly* is 25 of the 100 marks, ecosystem maturity directly reduces implementation risk.
- **Qwen's small dense models punch above their parameter count.** Public benchmarking (and Alibaba's own reporting) shows Qwen3-4B rivaling Qwen2.5-72B-Instruct on several tasks, and the 1.7B model was explicitly designed as a distillation target rather than a scaled-down afterthought — so it starts from a stronger instruction-following baseline than similarly-sized models from other families, meaning less fine-tuning effort is needed to reach production reliability.

### Hardware requirements

| Phase | Requirement | Notes |
|---|---|---|
| **Inference (bf16)** | ~3.5 GB VRAM | Fits comfortably on a single consumer GPU (e.g., RTX 3060/4060) |
| **Inference (4-bit / AWQ / GGUF)** | ~1–1.5 GB VRAM or CPU-only | Deployable on a laptop CPU or a small cloud instance; viable for edge/on-prem HR deployments |
| **LoRA fine-tuning (bf16)** | ~8–12 GB VRAM | Single mid-range GPU (RTX 3090/4090, T4, or an A10) is enough |
| **QLoRA fine-tuning (4-bit base + LoRA adapters)** | ~5–7 GB VRAM | Runs on a free-tier Colab T4 (16 GB) with headroom, or any 8 GB+ GPU |
| **Throughput** | High | Small dense model → high tokens/sec, low latency, cheap to batch for high-volume ATS ingestion |

This is the practical payoff of staying in the 0.5B–3B band: the entire training + inference lifecycle fits on hardware a single engineer can access (a laptop GPU or a free Colab tier), instead of requiring a multi-GPU cluster.

### Pros

- Best-in-class instruction-following and structured-output reliability *for its size class*, reducing how much fine-tuning is needed to hit high JSON-validity rates.
- Apache 2.0 — no commercial-use ambiguity, no need to negotiate a license for a paid HR product.
- 32K context comfortably covers resumes with room for schema + few-shot examples, without paying the latency/memory cost of a 128K–256K window we don't need.
- Non-thinking mode keeps outputs short and deterministic — important when the "output" *is* a JSON object, not prose.
- Mature, well-documented fine-tuning tooling (Unsloth/PEFT/TRL) lowers implementation risk for Part 3.
- Cheap enough to run on CPU at the edge (e.g., inside an on-prem HR system that can't send resumes to a third-party API for privacy/compliance reasons) — a genuine advantage over any hosted API.

### Cons

- At 1.7B, raw world knowledge and multi-step reasoning are weaker than 7B+ models — irrelevant for field extraction, but it means this model would be a poor choice if the platform later wants the *same* model to also write resume summaries, score candidates against a job description with nuanced judgment, or hold a multi-turn recruiting conversation.
- Being a dense (not MoE) small model, it doesn't have the "large-model knowledge, small-model cost" trick that MoE architectures offer — it genuinely only has 1.7B parameters of capacity, so its ceiling on any task requiring broad world knowledge is lower than similarly-*served* MoE models.
- Smaller community fine-tuning volume than Llama-family models (Llama still has more tutorials/blog posts in raw count), though Qwen3's own documentation and Unsloth's official notebooks close most of this gap.
- Base instruction-following, while strong for its size, will still need supervised fine-tuning to hit a high JSON-validity rate on messy real-world resumes (two-column PDFs converted to text, OCR artifacts, etc.) — it is a very good *starting point*, not a finished parser out of the box.

---

## 3. Why Not the Alternatives

### Qwen3-1.7B vs. **Llama 3.2 3B-Instruct**

| | Qwen3-1.7B | Llama 3.2 3B |
|---|---|---|
| Params | 1.7B | 3B |
| Context | 32K | 128K |
| License | Apache 2.0 | Llama 3.2 Community License (usage restrictions above 700M MAU; some jurisdictions/EU restrictions have applied) |
| Structured output | Trained with explicit JSON/tool-use objectives | Solid general instruction following, less consistently schema-tuned out of the box |
| Fine-tuning cost | Lower (smaller) | Slightly higher |

Llama 3.2 3B is a very capable model and a reasonable alternative — its 128K context is objectively larger, which would matter if we were parsing entire portfolios or multi-document candidate profiles. But for single-resume parsing, 32K is already more context than we need, and the Llama Community License carries commercial-use conditions (including restrictions tied to monthly active users and use in the EU for certain model versions) that add legal review overhead a resume-parsing product doesn't need to take on when a fully permissive Apache 2.0 model performs the same job. Given equal suitability for the actual task, Qwen3-1.7B wins on license simplicity and a smaller compute footprint.

### Qwen3-1.7B vs. **Phi-4-mini (3.8B)**

Phi-4-mini is MIT-licensed and Microsoft has specifically emphasized data quality over scale, with strong reasoning/math benchmarks for its size. It's a strong candidate and arguably the second choice here. It loses out for two reasons: (1) at 3.8B it sits above the "sweet spot" for cheap LoRA iteration compared to 1.7B, roughly doubling fine-tuning/serving cost for a task that doesn't need Phi-4-mini's reasoning strength (resume parsing is extraction, not multi-step math/logic); (2) Qwen3's dense line has more first-party emphasis on structured/tool-call output in its post-training recipe, which maps more directly onto "always return valid JSON matching this schema" than Phi-4-mini's reasoning-centric training.

### Qwen3-1.7B vs. **Gemma 3 (1B / 4B)**

Gemma 3 is Apache-friendly in spirit but ships under Google's custom Gemma license (not pure Apache 2.0/MIT), which includes an acceptable-use policy layered on top of the weights — more legal surface area than Qwen3's Apache 2.0. Technically, Gemma 3's 1B variant is closer in spirit to Qwen3-0.6B (arguably too small for reliable multi-field JSON extraction from noisy input), while the 4B variant lands past Qwen3-1.7B without a clear accuracy advantage for a narrow extraction task. Gemma 3 is an excellent choice for the "runs on literally any device" edge case, but Qwen3-1.7B remains the better default for a product that needs predictable licensing and strong structured-output behavior first.

### Why not go bigger (Qwen3-4B / 8B, or a 7B-class model generally)?

The temptation with any parsing task is "bigger = fewer errors." That's often true, but the marginal accuracy gain from 1.7B → 4B/8B on a **narrow, fine-tuned extraction task** is small relative to the 2–5x increase in fine-tuning time, VRAM, and serving cost — and this assessment is explicitly testing whether you can make a *small* model reliable through good data and training, not whether you can throw compute at the problem. If, after building and evaluating the 1.7B fine-tune, the JSON-validity or field-F1 numbers show a real ceiling (e.g., persistent failure on dense multi-column layouts), the natural escalation path is Qwen3-4B — same tokenizer, same license, same tooling, just a larger checkpoint — which keeps the whole pipeline (dataset, training script, eval harness) reusable.

### Why not the API route (GPT-4 Mini / hosted frontier models)?

This is the premise being replaced, so it's worth stating directly: a hosted API gives strong zero-shot accuracy with no fine-tuning effort, but (a) sends every candidate's PII to a third party, which is a real compliance concern for an HR platform, (b) has per-token cost that scales linearly with resume volume forever, with no way to buy down that cost through training the way you can with an owned model, and (c) can't be inspected, versioned, or regression-tested the way an owned checkpoint can — three things Part 4 of this assessment (evaluation, regression testing, version comparison) specifically depends on.

---

## 4. Other Models Considered — and When You'd Actually Use Them

The models below all came up during research as legitimate open-source options. None of them are "wrong" — each is a better fit for a *different* set of constraints than the ones this assessment optimizes for (small, cheap to fine-tune, permissively licensed, extraction-focused). Listing them, and being explicit about *when each one would be the right call*, is part of showing the reasoning the assessment is asking for.

| Model | Params | License | Where it beats Qwen3-1.7B / when you'd pick it |
|---|---|---|---|
| **Qwen3-0.6B** | 0.6B | Apache 2.0 | You need the absolute lowest inference cost (pure edge device, phone, Raspberry Pi) and can accept a lower JSON-validity ceiling after fine-tuning. Good first experiment to prove the pipeline before spending compute on the 1.7B run. |
| **Qwen3-4B** | 4B | Apache 2.0 | The 1.7B fine-tune plateaus below your target accuracy (e.g., struggles with dense multi-column resumes). Same tokenizer/tooling/license as 1.7B, so it's a drop-in upgrade — natural "next step" model. |
| **Llama 3.2 3B-Instruct** | 3B | Llama 3.2 Community License | You need the larger 128K context (e.g., parsing multi-document candidate packets, not single resumes), or your org already standardizes on the Llama ecosystem/tooling and can accept the license's usage conditions. |
| **Llama 3.2 1B-Instruct** | 1B | Llama 3.2 Community License | Similar niche to Qwen3-0.6B — ultra-lightweight edge deployment — if you're already committed to the Llama family for other parts of the stack. |
| **Phi-4-mini** | 3.8B | MIT | The parsing task grows to include judgment calls — e.g., inferring seniority level from vague job titles, or reasoning about date-range overlaps — where Phi-4-mini's stronger reasoning/math training pays off. MIT license is even simpler than Apache 2.0 if that matters to your legal team. |
| **Gemma 3 (1B / 4B)** | 1B–4B | Gemma license (custom, Apache-like but with an added usage policy) | You want Google's ecosystem (Vertex AI, native Gemini Nano tooling) or need Gemma 3's on-device optimization work (it's explicitly tuned for phones/laptops) more than you need Apache 2.0's simplicity. |
| **SmolLM3-3B** | 3B | Apache 2.0 | You want a model whose entire training *data and recipe* is published (not just weights) — useful if your organization has strict "fully open" auditability requirements, e.g., for a regulated HR/compliance context. |
| **Ministral-3-3B** | 3.4B (+0.4B vision) | Mistral commercial-friendly license | Resumes arrive as scanned images/screenshots rather than clean text and you want one model to handle both OCR-adjacent vision understanding *and* the extraction, instead of a separate OCR step. |
| **Mistral 7B / Ministral 8B** | 7–8B | Apache 2.0 / Mistral license | You've outgrown the 0.5–4B band entirely and have GPU budget for an 8B-class model; strong general baseline if resume parsing becomes one of several tasks the same model needs to handle well. |
| **Qwen2.5-1.5B / 3B (previous generation)** | 1.5B / 3B | Apache 2.0 | Qwen3 weren't available yet, or you specifically want Qwen2.5's slightly different training mix — it's still a very strong, well-documented small-model baseline and shows up repeatedly in fine-tuning tutorials, so it's a reasonable fallback if you hit tooling issues with Qwen3. |
| **GPT-4 Mini (hosted, closed)** | Unknown | Proprietary API | The one being replaced — still the right call for a *prototype/MVP* where you need to validate product-market fit before investing in fine-tuning infrastructure at all, since it needs zero training data or GPU setup. Not viable long-term due to per-token cost at scale and sending candidate PII to a third party. |

**Reading the table as a decision tree, roughly:**
- Need it to just work today, no ML engineering effort → **GPT-4 Mini** (until cost/privacy forces a change).
- Need it small, cheap, structured-output-first, and easy to fine-tune → **Qwen3-1.7B** (the pick).
- 1.7B isn't accurate enough after fine-tuning → **Qwen3-4B** (same family, scale up).
- Need huge context or already standardized on Meta's stack → **Llama 3.2 3B**.
- Task grows to need real reasoning/judgment, not just extraction → **Phi-4-mini**.
- Resumes arrive as images/scans → **Ministral-3-3B** (vision-capable).
- Need fully auditable open training data (compliance-heavy org) → **SmolLM3-3B**.

## 5. Summary Decision

**Qwen3-1.7B, non-thinking mode, fine-tuned via QLoRA, Apache 2.0 licensed**, is the selected model because it is the smallest checkpoint that plausibly starts with strong-enough instruction-following and structured-output behavior to become a reliable resume parser after a modest amount of supervised fine-tuning — while keeping every stage of the pipeline (training, inference, iteration) cheap enough to run on a single consumer GPU or a free-tier Colab instance, and licensed simply enough to ship in a commercial product without legal review overhead.
