#!/usr/bin/env python3
"""Prepare attributed, deduplicated Dolly instructions for controlled experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from checkpoints import atomic_json

REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
URL = f"https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/{REVISION}/databricks-dolly-15k.jsonl"
CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing"}


def prepare(rows):
    """No context-dependent questions, answers, truncation, or exact-normalized duplicates."""
    selected, seen = [], set()
    excluded = Counter()
    for index, row in enumerate(rows):
        if row.get("category") not in CATEGORIES or row.get("context", "").strip():
            excluded["category_or_context"] += 1
            continue
        text = row["instruction"].strip()
        if not 40 <= len(text) <= 1000:
            excluded["length"] += 1
            continue
        key = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if key in seen:
            excluded["normalized_duplicate"] += 1
            continue
        seen.add(key)
        selected.append({"text": text, "category": row["category"], "source_row": index})
    return selected, dict(excluded)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-file", type=Path, help="optional offline copy; pinned SHA-256 verified"
    )
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    args = p.parse_args(argv)
    if args.source_file:
        data = args.source_file.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dolly source differs from the pinned dataset hash")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    selected, excluded = prepare(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
    (args.out / "prompts.jsonl").write_bytes(payload)
    atomic_json(
        args.out / "provenance.json",
        {
            "dataset": "databricks/databricks-dolly-15k",
            "revision": REVISION,
            "source_url": URL,
            "source_sha256": SOURCE_SHA256,
            "license": "CC-BY-SA-3.0",
            "attribution": "Databricks, databricks-dolly-15k contributors",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "changes": "Selected context-free instructions in four categories, length 40-1000 characters; stripped surrounding whitespace and removed NFKC/case/whitespace-normalized duplicates. Human answers are excluded. No near-duplicate semantic filtering.",
            "input_rows": len(rows),
            "selected_rows": len(selected),
            "excluded": excluded,
            "categories": dict(Counter(row["category"] for row in selected)),
            "prompts_sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    (args.out / "ATTRIBUTION.md").write_text(
        "# Prompt data attribution\n\nInstructions are derived from Databricks databricks-dolly-15k, "
        "licensed under CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/ .\n\n"
        "Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k .\n\n"
        "The filtered prompt dataset retains that license. See provenance.json for the pinned "
        "revision, exact changes, filtering counts, and hashes. Human reference answers are not included.\n",
        encoding="utf-8",
    )
    print(f"prepared {len(selected)} unique prompts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
