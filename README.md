# Logcat Intelligence Engine

**A self-improving AI diagnostic system for Android OS logs** &mdash; a ReAct agent
diagnoses root causes from logcat/dmesg artifacts using specialized tools,
every investigation becomes training data, a local 7B model gets fine-tuned
with QLoRA + DPO, an eval harness gates deployment on accuracy, and the model
serves via an airgapped Docker image. New investigations feed back into
training &mdash; the flywheel is the actual product, not the model.

**[Live Demo &rarr;](https://satyamkumar02.github.io/logcat-intelligence-engine/)**
&mdash; a static showcase (architecture diagram, real recorded investigation
replay, real eval results, phase status). No backend, no live agent &mdash;
built from this repo's actual captured data, not staged content.

Built entirely on the **$0 stack**: Ollama (local `qwen2.5:7b`) as the agent
backbone, FAISS for RAG, QLoRA for fine-tuning, Groq's free tier for
distillation, Kaggle's free T4 for GPU training, vLLM + Docker for airgapped
deployment.

## What's in this repo

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full system diagram, tech stack rationale, AI/ML concepts glossary, interview talking points |
| [`docs/components/`](docs/components/) | One deep-dive per subsystem (agent, parsers, RAG, training pipeline, fine-tuning, eval harness, deployment, flywheel) |
| [`CONTEXT.md`](CONTEXT.md) | Live project status &mdash; what's done, what's verified, what's next |
| [`docs/EVAL_SET_IMPROVEMENT_PLAN.md`](docs/EVAL_SET_IMPROVEMENT_PLAN.md) | Tracked (not yet executed) plan to fix an eval-contamination issue found in Phase 3 |
| [`site/`](site/) | Source for the live demo above |

## Status

All 6 capstone phases have code written; most are locally verified end to
end. The one phase requiring cloud GPU access (Phase 4: QLoRA/DPO
fine-tuning) has its code written and locally verified wherever possible
without a GPU, but needs an actual Kaggle run to execute. See `CONTEXT.md`
for the full phase-by-phase status and the honest caveats behind every
"verified" claim (including why the 90% baseline eval accuracy is not a fair
zero-shot difficulty measure).

## Quickstart (local dev)

```bash
brew install ollama
ollama pull qwen2.5:7b

python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # fill in HF_TOKEN / WANDB_API_KEY / GROQ_API_KEY as needed
python scripts/validate_env.py
python scripts/generate_synthetic_logs.py
PYTHONPATH=. python scripts/run_agent_demo.py
```

See `docs/ARCHITECTURE.md` for everything else.
