"""Example client for the deployed diagnostic server.

The point of this file: it's nearly identical to how DiagnosticAgent talks
to Ollama in development (src/agent/diagnostic_agent.py) -- same `openai`
client, same chat.completions.create() call. Swapping --base-url from
Ollama to a deployed vLLM server is the entire "promotion to production"
step; no agent code changes. See docs/ARCHITECTURE.md's tech-stack table
for why this dev/prod parity was a deliberate design choice.

Usage (against a deployed server):
    python -m src.serve.client --base-url http://localhost:8000/v1 --model diagnostic-v1

Usage (against local Ollama, to prove the exact same code path works):
    python -m src.serve.client --base-url http://localhost:11434/v1 --model qwen2.5:7b
"""

from __future__ import annotations

import argparse

from openai import OpenAI


def diagnose(base_url: str, model: str, description: str) -> str:
    """Send one diagnostic prompt to an OpenAI-compatible server and return the reply.

    Args:
        base_url: OpenAI-API-prefixed URL (e.g. "http://localhost:8000/v1").
        model: Model identifier to request.
        description: A short problem description to diagnose.

    Returns:
        The model's raw text response.
    """
    client = OpenAI(base_url=base_url, api_key="none")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an Android diagnostic engineer."},
            {"role": "user", "content": description},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="diagnostic-v1")
    parser.add_argument(
        "--description",
        default=(
            "Diagnose this logcat: E AndroidRuntime: FATAL EXCEPTION: main\n"
            "E AndroidRuntime: java.lang.NullPointerException"
        ),
    )
    args = parser.parse_args()

    reply = diagnose(args.base_url, args.model, args.description)
    print(reply)


if __name__ == "__main__":
    main()
