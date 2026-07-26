# Logcat Intelligence Engine — Project Context

> Living reference file. Update this at the end of every work session so a new
> conversation can pick up state without re-reading the full spec docs.

## Goal

Build a self-improving AI diagnostic system for Android OS logs: a ReAct agent
diagnoses root causes from logcat/dmesg/bugreport artifacts using specialized
tools, every investigation is logged as a structured trace, traces become SFT +
DPO training data, a local 7B model (Qwen2.5-7B-Instruct) is fine-tuned with
QLoRA on that data, an eval harness gates deployment on accuracy, and the model
is served via vLLM inside an airgapped Docker container. New investigations
feed back into training — the flywheel is the actual product, not the model.

Source spec docs (read-only reference, not modified):
- `/Users/satyamkumar/Documents/learning/system-design-roadmap-new/ai-learning/capstone-project.md`
- `/Users/satyamkumar/Documents/learning/system-design-roadmap-new/ai-learning/capstone-build-requirements.md`

Working plan for the current build pass:
`/Users/satyamkumar/.claude/plans/here-is-the-project-delightful-patterson.md`

**Explain this project using**: `docs/ARCHITECTURE.md` (full system diagram,
tech stack rationale, AI/ML concepts glossary, interview talking points) plus
one deep-dive per subsystem in `docs/components/01-08` (diagnostic agent,
parsers, RAG retrieval, training data pipeline, fine-tuning/QLoRA/DPO/
distillation, eval harness, deployment/vLLM, flywheel) — each with a mermaid
diagram and an "IMPLEMENTED" vs "PLANNED" status.

## Cost Strategy — Target: $0

| Component | Free choice |
|---|---|
| Agent LLM (Phase 1) | Ollama `qwen2.5:7b` via OpenAI-compatible client (`OPENAI_BASE_URL=http://localhost:11434/v1`) |
| LLM Judge (Phase 3) | Ollama, local |
| Distillation teacher (Phase 4) | Groq free tier, Llama 3.3 70B |
| GPU training (Phase 4) | Kaggle Notebooks, free T4 (30 hr/week) |
| Experiment tracking | WandB free tier |
| Model weights | Qwen2.5-7B-Instruct from HuggingFace (free) |
| Deployment test | Docker locally (smoke test only); real GPU inference on Kaggle/RunPod |

Manual account setup required from the user (not automatable):
- [ ] HuggingFace account → `HF_TOKEN`
- [ ] WandB account → `WANDB_API_KEY`
- [ ] Groq console account → `GROQ_API_KEY` (needed at Phase 4)
- [ ] Kaggle account, GPU enabled in notebook settings (needed at Phase 4)

## Hardware

Apple M5, 24 GB RAM, arm64, ~800 GB free disk. Phases 0–3 run entirely
locally (no GPU needed). Phases 4–6 need CUDA (`bitsandbytes`, `vllm` don't
support Apple Silicon) — those steps run on Kaggle's free T4 GPU notebooks.

## Phase Status

| Phase | Status | Note |
|---|---|---|
| 0 — Environment setup | done | Ollama (`qwen2.5:7b` pulled), venv (Python 3.11), folder structure, git init, validate_env.py passing |
| 1 — Agentic diagnosis engine | done | Parsers, tools (incl. FAISS RAG), ReAct DiagnosticAgent verified end-to-end against local Ollama on 3 synthetic scenarios |
| 2 — Training data pipeline | done | TraceRecorder/Converter, DPOPairGenerator, dedup+quality filter, stats — verified end-to-end on 20 real traces (7 train / 2 val SFT, 1 DPO demo pair) |
| 3 — Eval harness | done | DiagnosticEval, TrajectoryGrader, LLMJudge, 10 labeled tasks — baseline run: 90% category accuracy (see caveats in decisions log) |
| 4 — Fine-tuning (cloud GPU) | not started | QLoRA SFT + DPO on Kaggle T4; Groq distillation teacher |
| 5 — Deployment | not started | Airgapped Docker + vLLM; local smoke test, real test on cloud |
| 6 — Flywheel | not started | Auto-retrain trigger, regression gate, version tracking |
| Stretch — Real AOSP data | not started | Deferred until synthetic pipeline is proven end-to-end |

## Key Decisions Log

- **2026-07-26**: Chose the $0 stack (Ollama + Groq free tier + Kaggle free T4 + HF/WandB free tiers) per user's explicit ask for free-cost build.
- **2026-07-26**: The doc's sample-data download URL (`storage.googleapis.com/android-logs-public/...`) is a placeholder and doesn't resolve. Decision: build a synthetic log generator first covering all 10 eval categories (ANR, crash, OOM, GPU fault, kernel panic, thermal, binder failure, camera crash, lowmemorykiller, memory leak); pulling real AOSP/CTS bugreports is a later stretch phase, not a blocker.
- **2026-07-26**: `requirements.txt` drops `bitsandbytes` and `vllm` (CUDA-only) for local Mac dev; a separate `requirements.cloud.txt` holds those for use in Kaggle/Colab notebooks during Phase 4/5.
- **2026-07-26**: `RAGRetrieverTool`'s FAISS index is bootstrapped from a small set of hand-written seed cases (one per eval category) so retrieval returns something meaningful before real trace data accumulates.
- **2026-07-26**: Phase 0 + Phase 1 completed in one pass. `qwen2.5:7b` pulled via Ollama, full ReAct loop (`DiagnosticAgent`) verified end-to-end against 3 synthetic scenarios (crash, ANR, GPU fault). Crash and ANR scenarios produced correct, well-evidenced diagnoses using all 4 tools (logcat_parser, pattern_search, dmesg_parser, rag_retriever). The GPU-fault run hit the known regex-based ReAct parsing fragility (model deviated from strict Thought/Action format mid-investigation, agent hit max-steps fallback) — this is expected zero-shot baseline behavior per the capstone doc (~30% zero-shot accuracy target), not a bug; Phase 3's eval harness will quantify this and Phase 4 fine-tuning is meant to close the gap.
- **2026-07-26**: `.gitignore` excludes generated artifacts under `data/processed/` (`*.faiss`, `case_metadata.jsonl`) but keeps `data/processed/seed_cases.jsonl` tracked since it's hand-written source data, not a build artifact.
- **2026-07-26**: Phase 2 completed. `scripts/collect_traces.py` ran the agent across all 10 synthetic categories (2 reps each, 20 investigations) against local Ollama, recording every trace via `TraceRecorder`. `scripts/build_training_data.py` deduped, quality-filtered, converted to ShareGPT SFT format, split train/val, and generated one DPO demo pair. Result: 9 of 20 traces survived dedup+filtering, split 7 train / 2 val, avg confidence 0.91, 7 distinct root-cause categories represented.
- **2026-07-26**: Two real bugs found and fixed while running Phase 2 at realistic volume (20 investigations, not the earlier 3-scenario demo): (1) `DiagnosticAgent._execute_tool()` crashed with an unhandled `TypeError` when the model emitted a malformed/empty tool `Action Input` — fixed by wrapping `tool(**args)` in a try/except (see `docs/components/01-diagnostic-agent.md` §8b). (2) `dedup_and_filter()` originally kept whichever occurrence of a repeated investigation appeared *first*, so a failed first run permanently shadowed a later successful re-run of the same scenario — fixed to keep the highest-confidence occurrence per dedup group (see `docs/components/04-training-data-pipeline.md` §8). Both fixes are already applied in `src/agent/diagnostic_agent.py` and `src/training/dedup.py`.
- **2026-07-26**: `data/sft/*.jsonl` and `data/dpo/train.jsonl` (small, curated) are committed to git as portfolio evidence; `data/raw/traces.jsonl` (large, ever-growing, unfiltered source) and the true scratch intermediates (`data/processed/traces_deduped.jsonl`, `data/processed/sft_all.jsonl`) stay gitignored.
- **2026-07-26**: Phase 3 completed. `scripts/run_eval.py` ran `DiagnosticEval` (outcome grading + `TrajectoryGrader` + `LLMJudge`) against the zero-shot `qwen2.5:7b` agent on all 10 labeled tasks in `data/eval/tasks.py`. Baseline result: **90% category accuracy** (9/10), avg keyword hit rate 0.87, avg trajectory score 1.00 (perfect — see caveat below), avg judge score 8.4/10. Reports written to `eval_results_baseline.md`/`.json` (gitignored, regenerable).
- **2026-07-26**: Important caveat on the 90% baseline — it is NOT comparable to the capstone doc's ~30% zero-shot expectation, and this was root-caused rather than reported uncritically. Two contamination sources identified: (1) the eval snippets contain near-literal category keywords (e.g. the literal string "GPU fault" in the gpu_fault task), making category recognition close to keyword matching rather than hard reasoning; (2) `data/processed/seed_cases.jsonl` (RAG corpus) and `data/eval/tasks.py` were both hand-written by the same author from the same category list at the same time — e.g. eval_001's description is a near-paraphrase of `seed_anr_01`'s — so RAG retrieval was handing back near-answers. The one miss (`eval_010`, memory_leak predicted as oom) is a genuinely defensible ambiguity, not a bug. Full writeup in `docs/components/06-eval-harness.md` §7. Conclusion: the harness itself is fully functional and ready to gate Phase 4, but a fair before/after fine-tuning comparison will need either real AOSP data (the stretch phase) or a new eval set built independently from the seed corpus — the current 10 tasks are an easy case, not a representative one.

## What's Next

Phase 0, 1, 2, and 3 are done and verified. Next up is Phase 4 (Fine-Tuning —
**requires cloud GPU**, first phase that can't run on this Mac): QLoRA SFT +
DPO training on Qwen2.5-7B via a free Kaggle T4 notebook, using the
`data/sft/`/`data/dpo/` datasets from Phase 2, with Groq's free Llama 3.3 70B
as the distillation teacher. Before investing in fine-tuning, consider
whether to first build a harder/independent eval set (see the Phase 3
caveat above) so the fine-tuning accuracy delta is actually meaningful.
