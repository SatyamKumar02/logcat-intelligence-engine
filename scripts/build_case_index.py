"""Build the FAISS seed index used by RAGRetrieverTool from hand-written seed cases.

Bootstraps retrieval with one case per eval category so the RAGRetrieverTool
has something meaningful to return before real investigation traces
accumulate. Re-run this whenever data/processed/seed_cases.jsonl changes, or
once real traces are appended to the case corpus in Phase 2.

Usage:
    python scripts/build_case_index.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_index(
    cases_path: Path,
    index_output_path: Path,
    metadata_output_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> int:
    """Embed case descriptions and write a FAISS index + metadata JSONL.

    Args:
        cases_path: Input JSONL with one case dict per line (must have
            a 'description' field used for the embedding).
        index_output_path: Where to write the FAISS index file.
        metadata_output_path: Where to write the metadata JSONL (same
            order as vectors in the index, so index i maps to line i).
        model_name: SentenceTransformer model to use for embeddings.

    Returns:
        Number of cases indexed.
    """
    cases = []
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    model = SentenceTransformer(model_name)
    descriptions = [c["description"] for c in cases]
    embeddings = model.encode(descriptions, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_output_path))

    with open(metadata_output_path, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-path", default="data/processed/seed_cases.jsonl")
    parser.add_argument("--index-output", default="data/processed/case_index.faiss")
    parser.add_argument("--metadata-output", default="data/processed/case_metadata.jsonl")
    args = parser.parse_args()

    count = build_index(Path(args.cases_path), Path(args.index_output), Path(args.metadata_output))
    print(f"Indexed {count} cases -> {args.index_output} / {args.metadata_output}")


if __name__ == "__main__":
    main()
