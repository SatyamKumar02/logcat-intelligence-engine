"""Health and readiness checks for a deployed OpenAI-compatible inference server.

Two independent checks, deliberately not collapsed into one:
- check_health(): is the process up at all? (matches vLLM's own /health
  endpoint and Docker's HEALTHCHECK directive in docker/Dockerfile.serve)
- check_ready(): does the model actually respond correctly to a real
  request? A process can be "up" (health check passes) while still loading
  a 7B model into GPU memory, which is not instantaneous.

Works against any OpenAI-compatible backend -- vLLM in production, or local
Ollama in development (see src/serve/client.py for the same dev/prod
parity argument applied to a usage example instead of a health check).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from openai import OpenAI


def check_health(base_url: str, timeout: float = 5.0) -> bool:
    """Check whether the server process is up via its /health endpoint.

    Args:
        base_url: Server root URL, e.g. "http://localhost:8000" (NOT the
            "/v1" OpenAI-API-prefixed URL used for chat completions).
        timeout: Request timeout in seconds.

    Returns:
        True if /health responded with HTTP 200, False otherwise (including
        on any connection error -- this never raises).
    """
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


@dataclass
class ReadinessResult:
    """Result of a live-completion readiness probe.

    Attributes:
        ready: Whether the model responded successfully.
        latency_ms: Round-trip time for the test completion, if it succeeded.
        error: Error message if the probe failed.
    """

    ready: bool
    latency_ms: float | None
    error: str | None = None


def check_ready(base_url: str, model: str, api_key: str = "none", timeout: float = 30.0) -> ReadinessResult:
    """Send a minimal real chat completion to confirm the model actually responds.

    Args:
        base_url: OpenAI-API-prefixed URL, e.g. "http://localhost:8000/v1".
        model: Model identifier to request.
        api_key: API key (vLLM/Ollama ignore the value but the client requires one).
        timeout: Request timeout in seconds.

    Returns:
        ReadinessResult with success/failure and latency.
    """
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    start = time.time()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=5,
            temperature=0.0,
        )
        return ReadinessResult(ready=True, latency_ms=(time.time() - start) * 1000)
    except Exception as e:
        return ReadinessResult(ready=False, latency_ms=None, error=str(e))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server root URL (no /v1 suffix)")
    parser.add_argument("--model", default="diagnostic-v1", help="Model identifier for the readiness probe")
    args = parser.parse_args()

    healthy = check_health(args.base_url)
    print(f"Health check ({args.base_url}/health): {'OK' if healthy else 'FAILED'}")

    readiness = check_ready(f"{args.base_url}/v1", args.model)
    if readiness.ready:
        print(f"Readiness check: OK ({readiness.latency_ms:.0f} ms)")
    else:
        print(f"Readiness check: FAILED ({readiness.error})")


if __name__ == "__main__":
    main()
