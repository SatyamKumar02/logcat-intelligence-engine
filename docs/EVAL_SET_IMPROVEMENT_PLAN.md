# Eval Set Improvement Plan

> Actionable instructions for fixing the eval-contamination problem found in
> Phase 3, to be done before trusting any Phase 4 (or later) fine-tuning
> accuracy comparison. Not yet started — this is a plan, not a report.

## The Problem, In Brief

`scripts/run_eval.py` measured **90% category accuracy** on the zero-shot
`qwen2.5:7b` agent — far above the capstone spec's ~30% expectation. This
is not because the agent is unusually good; it's because the eval set has
two contamination sources (full analysis in
[`components/06-eval-harness.md`](components/06-eval-harness.md) §7):

1. **Eval snippets contain near-literal category keywords** (e.g. the exact
   string `"GPU fault"` in the gpu_fault task) — recognizing the category is
   close to keyword matching, not hard reasoning over a noisy real log.
2. **RAG/eval leakage** — `data/processed/seed_cases.jsonl` (the RAG corpus)
   and `data/eval/tasks.py` (the eval set) were both hand-written by the
   same author, from the same category list, at the same time. Several task
   descriptions are near-paraphrases of their matching seed case
   description, so `rag_retriever` often hands the agent a near-answer.

**Consequence:** a before/after fine-tuning accuracy comparison run against
today's eval set would not be trustworthy — a ceiling effect this close to
100% has little room to show improvement, and any change (up or down) is as
likely to reflect eval quirks as real model quality.

## Why This Matters Before Phase 4 Conclusions

Fine-tuning's entire value proposition in this project is a measured
accuracy delta (target: +15-25pp per `capstone-project.md`). Without a fair
eval set, that number cannot be trusted in either direction — a fine-tuned
model could look identical to baseline (both near the keyword-matching
ceiling) even if it genuinely reasons better, or could look worse due to
noise in a near-saturated metric. Fix the measurement before trusting what
it measures.

## The Fix: A Two-Tier Eval Strategy

Keep the current 10 tasks as a **fast regression smoke test** (cheap, catches
gross regressions), and add a **held-out, independently-sourced eval set**
as the real benchmark for fine-tuning decisions.

### Step 1 — Source eval cases independently from the RAG seed corpus

Pick one (or combine):

- **Real AOSP/CTS data** (ties into `CONTEXT.md`'s "Stretch — Real AOSP
  data" phase) — pull actual public bugreports/logcat samples and hand-label
  their root cause. Real bugreports are noisy (thousands of unrelated
  lines), which directly fixes contamination source #1 above.
- **Independently-generated synthetic cases** — write a *second* batch of
  synthetic scenarios (or have a different, blind process generate them)
  without referencing `seed_cases.jsonl`'s wording. Vary phrasing
  deliberately from the seed corpus's descriptions.
- **Held-out seed cases** — split the seed corpus itself: keep half the
  hand-written cases in the RAG index, and turn the *other* half into eval
  tasks that were never indexed. This is the cheapest option and doesn't
  require new data authoring, at the cost of a smaller eval set.

### Step 2 — Ablate RAG to quantify the leakage directly

Before/alongside building new data, run a diagnostic experiment on the
*current* eval set: re-run `scripts/run_eval.py` with `rag_tool=None` (drop
the `--` — just don't build the FAISS index, or pass `rag_tool=None`
directly to `DiagnosticAgent`) and compare accuracy with RAG on vs. off.
A large accuracy drop with RAG disabled would directly confirm how much of
the 90% is retrieval leakage vs. genuine reasoning — cheap to run (no new
data needed), and a good number to cite either way.

### Step 3 — Increase eval task count

10 tasks means each one is worth 10 accuracy points — noisy at this scale.
Once new independent data exists, aim for 25-50+ tasks across the same
categories (multiple examples per category) for a more statistically
meaningful accuracy number.

### Step 4 — Re-baseline on the new eval set

Run `scripts/run_eval.py` (pointed at the new task set) against the
*current, un-fine-tuned* agent to get a trustworthy pre-fine-tuning number.
Expect this to land closer to the spec's ~30-50% range if the contamination
diagnosis above is correct — a low baseline is not a problem, it's the
whole point (see `docs/components/06-eval-harness.md`'s "Eval First, Train
Second" mental model).

### Step 5 — Re-run Phase 4 fine-tuning evaluation against the same clean set

Once Phase 4 produces a merged model (see
[`components/05-finetuning-pipeline.md`](components/05-finetuning-pipeline.md)),
run the same new eval set against both the base and fine-tuned model. *That*
delta is the number worth reporting as the project's headline metric —
not the current 90%.

## What NOT to Do

- Don't just discard the current 10 tasks — they're still useful as a fast
  smoke test (does a code change obviously break something) even though
  they're not a fair fine-tuning benchmark.
- Don't hand-tune the new eval set's difficulty to hit a target accuracy
  number — the goal is measurement validity, not a specific headline
  number.
- Don't skip Step 2 (the RAG ablation) even if new data takes a while to
  build — it's the single cheapest experiment that directly tests the
  leakage hypothesis with zero new data required.

## Status

Not started. Tracked here rather than in `CONTEXT.md`'s decision log because
this is forward-looking work to execute, not a decision already made.
