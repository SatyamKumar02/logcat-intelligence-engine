# Component: RAG Retrieval (Case Similarity Search)

**Status: IMPLEMENTED** — `src/agent/tools.py::RAGRetrieverTool`, `scripts/build_case_index.py`, `data/processed/seed_cases.jsonl`

Gives the agent access to "institutional memory" — similar past diagnostic
cases — so its answer can be grounded in precedent rather than reasoning from
the current log file alone.

---

## 1. Why RAG Here

The agent's first three tools (logcat/dmesg parsing, pattern search) only
ever look at the *current* artifact. But real diagnostic work leans heavily
on pattern recognition across many past incidents ("I've seen this GPU fault
signature before, it was always the Adreno driver"). RAG is how that
experience gets encoded computationally: past cases are embedded once,
indexed, and retrieved by semantic similarity to the current symptom
description — no fine-tuning required to add new institutional knowledge,
just append to the case corpus and rebuild the index.

---

## 2. Architecture

```mermaid
flowchart TD
    subgraph Offline["Offline: Index Build (scripts/build_case_index.py)"]
        SEED["data/processed/seed_cases.jsonl\n(10 hand-written cases,\none per eval category)"]
        EMB1["SentenceTransformer\n('all-MiniLM-L6-v2')\n.encode(descriptions)"]
        NORM1["normalize_embeddings=True\n(L2-normalized vectors)"]
        IDX["faiss.IndexFlatL2\n.add(embeddings)"]
        WRITE["faiss.write_index()\n-> case_index.faiss\n+ case_metadata.jsonl"]
        SEED --> EMB1 --> NORM1 --> IDX --> WRITE
    end

    subgraph Online["Online: Query Time (RAGRetrieverTool.__call__)"]
        QUERY["query string\n(from agent's Action Input)"]
        EMB2["SentenceTransformer\n.encode([query])"]
        SEARCH["index.search(embedding, top_k)"]
        SCORE["similarity_score = 1 - L2_distance"]
        RESULT["ToolResult:\nsimilar_cases[] with scores"]
        QUERY --> EMB2 --> SEARCH --> SCORE --> RESULT
    end

    WRITE -.faiss.read_index\nat RAGRetrieverTool.__init__.-> SEARCH
```

---

## 3. The Embedding Model

**`all-MiniLM-L6-v2`** (via `sentence-transformers`): a 6-layer distilled
transformer producing 384-dimensional sentence embeddings. Chosen because:

- It's free, runs on CPU in milliseconds — no GPU needed even for the
  embedding step, which matters since this tool runs inside the agent's
  synchronous ReAct loop (embedding latency adds directly to investigation
  time).
- ~80MB download, trivial to bundle/cache.
- Strong general-purpose semantic similarity performance for short
  descriptive text (exactly what a symptom description or case description
  is), even though it wasn't specifically trained on Android diagnostic text.

**Trade-off worth naming:** a domain-specific embedding model (e.g. one
fine-tuned on Android bug reports) would likely separate categories with
larger margins. `all-MiniLM-L6-v2` is the right *default* choice for a $0
build — swapping the model is a one-line change in `RAGRetrieverTool`'s
`model_name` parameter if retrieval quality becomes a bottleneck later.

---

## 4. Index: FAISS `IndexFlatL2`

`IndexFlatL2` does brute-force (exact) nearest-neighbor search by Euclidean
distance — no approximation, no training step required, just
`index.add(vectors)` then `index.search(query, k)`. This is the right choice
at the current corpus size (10 seed cases now, expected to grow to hundreds
as real traces accumulate in Phase 2) — exact search is fast enough and there
is zero index-quality risk. It would only need to change (to `IndexIVFFlat`
or `IndexHNSWFlat`) if the corpus grew into the tens-to-hundreds-of-thousands
range where brute-force search latency becomes noticeable.

**Why L2 distance is converted to a similarity score:**
```python
distances, indices = self._index.search(embedding, top_k)
...
case["similarity_score"] = float(1 - dist)
```
Because embeddings are L2-normalized (`normalize_embeddings=True`) before
indexing, for unit vectors `L2_distance² = 2 - 2·cosine_similarity`, so
smaller L2 distance means higher cosine similarity. The `1 - dist` conversion
here is an approximation for display purposes — it produces a monotonically
decreasing "score" as distance grows (higher = more similar) without needing
the exact cosine formula, which is good enough since the tool only uses the
score for ranking/display, not for a downstream numeric threshold decision.

---

## 5. Seed Corpus Bootstrap

`data/processed/seed_cases.jsonl` — 10 hand-written cases, one per eval
category (`anr`, `crash`, `oom`, `gpu_fault`, `oom_kill`, `thermal`,
`camera_crash`, `kernel_panic`, `binder_failure`, `memory_leak`). Each has:

```json
{
  "case_id": "seed_gpu_fault_01",
  "description": "Device reboots unexpectedly during video playback or 3D game with no app-level crash",
  "root_cause": "GPU command buffer fault or hang in the Adreno/kgsl driver during hardware-accelerated rendering",
  "root_cause_category": "gpu_fault",
  "recommended_action": "Capture a GPU trace, check for driver updates, file a bug against the GPU driver team"
}
```

**Why hand-written seeds instead of starting empty:** an empty index means
`RAGRetrieverTool` always returns nothing, which teaches the agent (and any
future SFT training on its traces) that RAG retrieval is useless — bad
signal to bake in early. Ten seed cases, one per category, guarantee at least
one relevant precedent exists for every eval task from day one. As real
investigation traces accumulate (Phase 2), the plan is to append verified,
high-confidence traces into this same corpus and rebuild the index —
`scripts/build_case_index.py` is written generically against any
`cases_path` JSONL with a `description` field, so this requires no code
changes later.

---

## 6. Verified Behavior

Manually tested query: `"device crashed unexpectedly during game, GPU issue suspected"` against the 10-seed index returned, in order:

1. `seed_gpu_fault_01` (score 0.322) — correct top match
2. `seed_camera_crash_01` (score -0.197)
3. `seed_oom_01` (score -0.282)

The GPU-fault case ranking first, with a clear score gap to the next
candidates, confirms the embed → index → search pipeline is behaving
sensibly even at this tiny corpus size.

---

## 7. How to Explain This Component in an Interview

- "RAG here isn't about answering general knowledge questions — it's
  retrieval over the team's own accumulated diagnostic history, which is
  exactly the kind of proprietary, ever-growing corpus that makes RAG worth
  the complexity over just prompting a bigger model."
- "I used exact search (`IndexFlatL2`) deliberately rather than reaching for
  an ANN index by default — at low-thousands of cases, exact search is both
  simpler and fast enough, and I'd only add approximate search once I could
  measure it actually being a bottleneck."
- "The retrieval corpus and the fine-tuning corpus are the same underlying
  data — verified traces — just used two different ways: as few-shot
  precedent at inference time (RAG) and as training signal offline (SFT).
  That's a nice property of designing the trace format once, upfront."
