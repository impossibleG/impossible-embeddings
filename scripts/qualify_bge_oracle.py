#!/usr/bin/env python3
"""Generate deterministic BGE Small qualification evidence from local pinned files.

This script is intentionally offline-only. Download and hash-check the artifacts named in the
curated manifest before running it, then pass the directory containing ``config.json``,
``model.safetensors``, and the tokenizer files. It never executes model repository code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from transformers import AutoModel, AutoTokenizer


REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EXPECTED_VERSIONS = {"torch": "2.8.0", "transformers": "4.56.1"}
CASES = (
    ("document_public", "document", "The weather is lovely today."),
    ("document_unicode", "document", "Olá, 世界 — naïve café 🚀"),
    ("query_public", "query", "What is the capital of France?"),
    ("document_empty", "document", ""),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_text(task: str, text: str) -> str:
    return f"{QUERY_PREFIX}{text}" if task == "query" else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    torch_version = torch.__version__.split("+")[0]
    versions = {"torch": torch_version, "transformers": transformers.__version__}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"qualification requires exact software versions {EXPECTED_VERSIONS}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    model = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model.eval()
    torch.set_grad_enabled(False)

    def encode(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        inputs = tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        output = model(**inputs).last_hidden_state[:, 0]
        normalized = torch.nn.functional.normalize(output, p=2, dim=1)
        token_ids = [
            row[: int(mask.sum().item())].tolist()
            for row, mask in zip(inputs["input_ids"], inputs["attention_mask"], strict=True)
        ]
        return token_ids, normalized.tolist()

    prepared = [prepared_text(task, text) for _, task, text in CASES]
    token_ids, batch_vectors = encode(prepared)
    evidence_cases: list[dict[str, Any]] = []
    for (name, task, text), ids, batch_vector, ready in zip(
        CASES, token_ids, batch_vectors, prepared, strict=True
    ):
        single_ids, single_vectors = encode([ready])
        max_delta = max(abs(left - right) for left, right in zip(batch_vector, single_vectors[0]))
        if ids != single_ids[0] or max_delta > 1e-6:
            raise RuntimeError(f"batch/single mismatch for {name}: {max_delta}")
        evidence_cases.append(
            {
                "name": name,
                "task": task,
                "text": text,
                "token_ids": ids,
                "embedding": [round(value, 9) for value in batch_vector],
            }
        )

    exact_limit_text = " ".join(["hello"] * 510)
    over_limit_text = f"{exact_limit_text} hello"
    exact_ids = tokenizer(exact_limit_text, add_special_tokens=True, truncation=False)["input_ids"]
    over_ids = tokenizer(over_limit_text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(exact_ids) != 512 or len(over_ids) != 513:
        raise RuntimeError("the public 512-token boundary fixture changed")

    artifacts = {}
    for name in (
        "config.json",
        "model.onnx",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = args.model_dir / name
        artifacts[name] = {"size": path.stat().st_size, "sha256": file_sha256(path)}

    evidence = {
        "schema_version": 1,
        "model": "BAAI/bge-small-en-v1.5",
        "revision": REVISION,
        "oracle": {
            "implementation": "transformers.AutoModel/BertModel CLS pooling and L2 normalization",
            "remote_code": False,
            "offline": True,
            "software": versions,
        },
        "artifacts": artifacts,
        "cases": evidence_cases,
        "limits": {
            "max_tokens_including_special_tokens": 512,
            "exact_fixture": {"token": "hello", "repetitions": 510, "token_count": 512},
            "over_fixture": {"token": "hello", "repetitions": 511, "token_count": 513},
        },
    }
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
