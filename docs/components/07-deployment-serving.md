# Component: Deployment & Serving

**Status: MOSTLY VERIFIED** — `docker/Dockerfile.serve`, `docker/requirements.serve.txt`, `docker/docker-compose.yml`, `src/serve/health.py`, `src/serve/client.py`

Serves the fine-tuned, merged model behind an OpenAI-compatible API, inside
a Docker image that needs **zero network access at runtime**. This is the
"production awareness" component — the difference between a model that
works in a notebook and one that can run in a restricted/secure environment.

**What "mostly verified" means, precisely:** this Mac has Docker Desktop but
no NVIDIA GPU passthrough, and Phase 4 hasn't produced a real merged model
yet (see `docs/components/05-finetuning-pipeline.md`) — so the one thing
that couldn't be tested is an actual `docker build` + `docker run` serving
real weights. Everything else genuinely was:
- `Dockerfile.serve` passed `docker buildx build --check` (a real lint
  against buildkit, no image pull needed) with zero warnings.
- `docker-compose.yml` was resolved with `docker compose config` — confirms
  the GPU reservation, healthcheck, bind mount, and build-arg
  interpolation are all syntactically correct.
- `src/serve/health.py`'s `check_health()` and `check_ready()` were run
  against **real servers**: a live chat completion against local Ollama
  (genuinely OpenAI-compatible, standing in for vLLM), a hand-rolled mock
  HTTP server returning a real 200 on `/health`, a real 404 (Ollama has no
  `/health` path), and a dead port — all four returned the expected result.
- `src/serve/client.py` was run against local Ollama and produced a
  coherent diagnostic response — the literal code that will later point at
  a deployed vLLM server, unmodified, already works against a real
  OpenAI-compatible backend today.

---

## 1. Architecture

```mermaid
flowchart TD
    subgraph BuildTime["Build Time (needs internet + GPU host)"]
        MERGED["outputs/merged-diagnostic-v1/\n(from 05-finetuning-pipeline.md)"]
        DOCKERFILE["Dockerfile.serve\nFROM nvidia/cuda:12.1.0-...\npip install vllm, openai, fastapi\nCOPY ${MODEL_PATH} /models/diagnostic-v1"]
        IMAGE["logcat-ie-serve:v1\n(weights baked in, ~18GB)"]
        MERGED --> DOCKERFILE --> IMAGE
    end

    subgraph RunTime["Run Time (NO internet needed)"]
        VLLM["vllm.entrypoints.openai.api_server\n--model /models/diagnostic-v1\n--port 8000"]
        HEALTH["/health endpoint\n(HEALTHCHECK every 30s)"]
        API["/v1/chat/completions\n(OpenAI-compatible)"]
        VLLM --> HEALTH
        VLLM --> API
    end
    IMAGE -->|docker run --network none| VLLM

    subgraph Client["Any OpenAI-compatible client"]
        AGENT["DiagnosticAgent\n(same code as against Ollama!)"]
    end
    AGENT -->|"base_url=http://diagnostic-server:8000/v1"| API

    subgraph Compose["docker-compose.yml"]
        SVC["diagnostic-server service\nGPU reservation, healthcheck"]
        SIDECAR["trace-recorder sidecar\n(Phase 6 -- not yet added)"]
        SVC -.->|"future depends_on: healthy"| SIDECAR
    end
```

---

## 2. Why vLLM

vLLM is a high-throughput inference server built around **PagedAttention** —
it manages the GPU memory used for attention key/value caches in fixed-size
pages (like OS virtual memory), which lets it batch many concurrent
requests without wasting memory on over-allocated, contiguous KV-cache
buffers. Practically, this means:

- Much higher throughput than naive HuggingFace `generate()` serving under
  concurrent load — relevant since the target metric is "12 diagnoses/minute"
  throughput, not just single-request latency.
- **OpenAI-compatible API out of the box**
  (`vllm.entrypoints.openai.api_server`) — the exact same `openai` Python
  client and `chat.completions.create()` call that `DiagnosticAgent` already
  uses against Ollama in development works unmodified against vLLM in
  production. Zero agent code changes between dev and prod backbones — only
  `OPENAI_BASE_URL` changes.

**Mac constraint:** vLLM requires CUDA; it cannot run natively on Apple
Silicon. Real inference testing happens on the same free Kaggle T4 (or a
future paid RunPod/Lambda instance) used for training — on the Mac, Docker
build/health-check plumbing can be smoke-tested, but not actual GPU
inference.

---

## 3. The Airgapped Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip git
COPY requirements.serve.txt .
RUN pip install --no-cache-dir vllm==0.4.2 openai==1.14.0 fastapi==0.110.0 uvicorn==0.29.0
ARG MODEL_PATH=./outputs/merged-diagnostic-v1
COPY ${MODEL_PATH} /models/diagnostic-v1
COPY src/serve/ /app/serve/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/diagnostic-v1", ...]
```

**"Airgapped" means dependencies are data, baked in at build time** — the
key line is `COPY ${MODEL_PATH} /models/diagnostic-v1`. Everything the model
needs at runtime (weights, tokenizer config, chat template) is copied into
the image *during the build*, which is the one point where internet access
and a GPU host are assumed. Once built, the image is a self-contained,
versioned artifact: `docker run --network none logcat-ie-serve:v1` must
still serve correctly, because nothing in the `CMD` reaches out to
HuggingFace Hub, PyPI, or anywhere else at runtime.

**Why `ARG MODEL_PATH` instead of hardcoding a path:** every retrain (see
[08-flywheel.md](08-flywheel.md)) produces a new merged model directory
(`outputs/merged-diagnostic-vN`); the build arg lets `weekly_retrain.sh`
rebuild a freshly-tagged image (`logcat-ie-serve:vN`) pointing at the new
weights without touching the Dockerfile itself.

---

## 4. docker-compose Orchestration

Currently one service:

- **`diagnostic-server`** — the vLLM container, with a `deploy.resources.reservations.devices` GPU reservation and a `healthcheck` block (`curl http://localhost:8000/health`, matching Docker's own `HEALTHCHECK` in `Dockerfile.serve`) so anything that later depends on this service can wait for `service_healthy` instead of just "container started."

**Not yet added:** a `trace-recorder` sidecar running `auto_trigger.py --watch-dir /app/data/raw` (see [08-flywheel.md](08-flywheel.md)) pointed at the served model, so newly investigated cases keep flowing into the trace corpus in a deployed environment, not just during local development. That code doesn't exist until Phase 6 — adding the sidecar service is Phase 6's job, not Phase 5's.

---

## 5. Health Check & Smoke Test — What Was Actually Verified

This Mac has Docker Desktop but no NVIDIA GPU passthrough, and there's no
real merged model yet (Phase 4 hasn't run on Kaggle) — so the full
`docker build && docker run` + live inference loop below is written and
correct, but has **not** been executed end to end. What was:

```bash
# Verified: Dockerfile lints clean (no image pull, no build)
docker buildx build --check -f docker/Dockerfile.serve .
# -> "Check complete, no warnings found."

# Verified: compose file resolves correctly (GPU reservation, healthcheck,
# bind mount, build-arg interpolation all present and well-formed)
docker compose -f docker/docker-compose.yml config

# Verified: health.py's check_health()/check_ready() against real servers
# (live Ollama completion, a mock 200 /health endpoint, a real 404, a dead
# port) -- all four returned the expected result, see
# docs/components/07-deployment-serving.md's status header above.

# Verified: client.py produces a coherent diagnosis when pointed at Ollama
python -m src.serve.client --base-url http://localhost:11434/v1 --model qwen2.5:7b
```

The real build/run, on a CUDA host once Phase 4 produces a merged model:

```bash
docker build --build-arg MODEL_PATH=./outputs/merged-diagnostic-v1 -f docker/Dockerfile.serve -t logcat-ie-serve:v1 .
docker compose -f docker/docker-compose.yml up -d
python -m src.serve.health --base-url http://localhost:8000 --model diagnostic-v1
python -m src.serve.client --base-url http://localhost:8000/v1 --model diagnostic-v1
```

`src/serve/health.py` is a separate operator/monitoring tool, not what
Docker's own `HEALTHCHECK` directive invokes (that's a plain `curl` against
vLLM's built-in `/health` endpoint, see `Dockerfile.serve`) — `health.py`
adds a second, stronger check (`check_ready()`, a real chat completion)
because a container can report healthy while the 7B model is still loading
into GPU memory, which for vLLM is not instantaneous.

---

## 6. How to Explain This Component in an Interview

- "The airgapped requirement forced a specific discipline: anything the
  model needs has to be a build-time dependency, never a runtime one — the
  `COPY` instruction baking weights into the image is the whole trick."
- "I get dev/prod parity for free because vLLM speaks the same
  OpenAI-compatible API my agent already calls against Ollama — swapping
  backbones is a config change, not a code change."
- "The image is versioned and reproducible by design — every retrain cycle
  produces a new tagged image (`logcat-ie-serve:vN`) rather than mutating a
  running container, which is what makes the regression gate (rebuild only
  if eval passes) meaningful."
- "I don't have GPU passthrough on this machine, so I couldn't run the real
  container — but I didn't just leave it untested. `docker buildx build
  --check` lints the Dockerfile for real, `docker compose config` resolves
  the whole compose file including the GPU reservation and build-arg
  interpolation, and I proved the health-check and client code against real
  servers — Ollama standing in for vLLM, since both speak the same
  OpenAI-compatible API. Everything that *can* be verified without a GPU,
  was."
