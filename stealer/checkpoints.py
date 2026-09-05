"""Atomic batch checkpoints and an exclusive writer lock for long experiments."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as fh:
            temp = Path(fh.name)
            json.dump(value, fh, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


@contextlib.contextmanager
def writer_lock(out):
    """OS-managed lock: released after interruption, including a killed process."""
    with (Path(out) / ".writer.lock").open("a+b") as fh:
        if os.name == "nt":
            import msvcrt

            fh.write(b"0")
            fh.flush()
            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("another process is using this experiment directory") from exc
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("another process is using this experiment directory") from exc
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


class Checkpoints:
    def __init__(self, out, manifest, resume=False):
        self.out = Path(out)
        path = self.out / "manifest.json"
        if resume:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != manifest:
                raise ValueError("resume requires identical config, prompts, and source files")
        else:
            if path.exists():
                raise ValueError("experiment already exists; use --resume")
            atomic_json(path, manifest)
        self.manifest = manifest
        (self.out / "batches").mkdir(exist_ok=True)

    def bind_runtime(self, runtime):
        path = self.out / "runtime.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != runtime:
                raise ValueError(
                    "runtime/model/tokenizer/device changed; cannot mix resumed batches"
                )
        else:
            atomic_json(path, runtime)

    def _path(self, split, start, arm):
        if split not in ("train", "calibration", "evaluation") or arm not in (
            "watermarked",
            "baseline",
            "control",
        ):
            raise ValueError("unknown split or arm")
        if start < 0:
            raise ValueError("batch start must be nonnegative")
        return self.out / "batches" / f"{split}-{start:08d}-{arm}.json"

    def _validate(self, rows, split, start, arm, prompts, seed):
        if len(rows) != len(prompts):
            raise ValueError("incomplete checkpoint batch")
        ctx = self.manifest["config"]["context_len"]
        max_tokens = self.manifest["config"]["max_new_tokens"]
        for i, (row, prompt) in enumerate(zip(rows, prompts, strict=True)):
            expected = {
                "id": f"{split}-{start + i}",
                "split": split,
                "arm": arm,
                "prompt": prompt,
                "seed": seed,
            }
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError("checkpoint row identity differs from the planned batch")
            ids = row.get("token_ids")
            if (
                not isinstance(ids, list)
                or len(ids) > max_tokens
                or any(type(t) is not int or t < 0 for t in ids)
            ):
                raise ValueError("invalid checkpoint token IDs")
            gs = row.get("g_values")
            if not isinstance(gs, list) or len(gs) != max(0, len(ids) - ctx):
                raise ValueError("invalid checkpoint g-values")

    def load(self, split, start, arm, prompts, seed):
        path = self._path(split, start, arm)
        if not path.exists():
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
        rows = envelope["rows"]
        if digest(rows) != envelope["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        self._validate(rows, split, start, arm, prompts, seed)
        return rows

    def save(self, rows, split, start, arm, prompts, seed):
        self._validate(rows, split, start, arm, prompts, seed)
        atomic_json(self._path(split, start, arm), {"rows": rows, "sha256": digest(rows)})

    def completed(self):
        path = self.out / "generation.json"
        if not path.exists():
            return None
        generation = json.loads(path.read_text(encoding="utf-8"))
        corpus = self.out / "corpus.jsonl"
        if file_digest(corpus) != generation["corpus_sha256"]:
            raise ValueError("completed corpus checksum mismatch")
        rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
        expected = 3 * sum(len(group) for group in self.manifest["splits"].values())
        if len(rows) != expected:
            raise ValueError("completed corpus row count mismatch")
        return rows, generation["metadata"]

    def finish(self, rows, metadata):
        expected = 3 * sum(len(group) for group in self.manifest["splits"].values())
        if len(rows) != expected:
            raise ValueError("cannot finalize an incomplete corpus")
        atomic_json(
            self.out / "generation.json",
            {"corpus_sha256": file_digest(self.out / "corpus.jsonl"), "metadata": metadata},
        )
