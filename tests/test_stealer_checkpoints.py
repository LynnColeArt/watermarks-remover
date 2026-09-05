"""Durability, protocol isolation, and offline reanalysis for long runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stealer"))

import checkpoints
import evaluate


def manifest():
    return {
        "config": {"context_len": 1, "max_new_tokens": 8},
        "splits": {"train": ["prompt"]},
        "source_sha256": {"x": "a"},
    }


def row():
    return {
        "id": "train-0",
        "split": "train",
        "arm": "baseline",
        "prompt": "prompt",
        "seed": 1,
        "token_ids": [1, 2, 3],
        "g_values": [0.5, 0.5],
        "reference_score": 0.5,
        "text": "one two three",
    }


def test_batch_survives_restart_and_rejects_tampering(tmp_path):
    store = checkpoints.Checkpoints(tmp_path, manifest())
    store.save([row()], "train", 0, "baseline", ["prompt"], 1)
    resumed = checkpoints.Checkpoints(tmp_path, manifest(), resume=True)
    assert resumed.load("train", 0, "baseline", ["prompt"], 1) == [row()]
    assert resumed.load("train", 0, "watermarked", ["prompt"], 1) is None
    path = next((tmp_path / "batches").glob("*.json"))
    data = json.loads(path.read_text())
    data["rows"][0]["token_ids"][0] = 9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        resumed.load("train", 0, "baseline", ["prompt"], 1)


def test_batch_identity_and_protocol_are_checked(tmp_path):
    store = checkpoints.Checkpoints(tmp_path, manifest())
    with pytest.raises(ValueError, match="identity"):
        store.save([row()], "train", 0, "baseline", ["changed"], 1)
    with pytest.raises(ValueError, match="incomplete"):
        store.save([], "train", 0, "baseline", ["prompt"], 1)
    changed = manifest()
    changed["config"]["context_len"] = 2
    with pytest.raises(ValueError, match="identical"):
        checkpoints.Checkpoints(tmp_path, changed, resume=True)
    store.bind_runtime({"device": "cuda", "torch": "one"})
    with pytest.raises(ValueError, match="runtime"):
        store.bind_runtime({"device": "cpu", "torch": "one"})


def test_atomic_write_preserves_previous_batch_after_failure(monkeypatch, tmp_path):
    path = tmp_path / "batch.json"
    checkpoints.atomic_json(path, {"committed": 1})

    def fail_replace(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr(checkpoints.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interruption"):
        checkpoints.atomic_json(path, {"committed": 2})
    assert json.loads(path.read_text()) == {"committed": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_writer_lock_excludes_competitors_and_releases(tmp_path):
    with (
        checkpoints.writer_lock(tmp_path),
        pytest.raises(ValueError, match="another process"),
        checkpoints.writer_lock(tmp_path),
    ):
        pytest.fail("competing writer acquired the lock")
    with checkpoints.writer_lock(tmp_path):
        pass


def test_complete_run_can_be_resumed_and_scored_without_generation(monkeypatch, tmp_path):
    calls = []

    def fake_generate(args, splits, out, checkpoint):
        calls.append(1)
        rows = []
        for split, prompts in splits.items():
            for i, prompt in enumerate(prompts):
                for arm in ("watermarked", "baseline", "control"):
                    rows.append(
                        {
                            **row(),
                            "id": f"{split}-{i}",
                            "split": split,
                            "arm": arm,
                            "prompt": prompt,
                        }
                    )
        (out / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rows, {"model": "fake", "dtype": "float32"}

    monkeypatch.setattr(evaluate, "generate", fake_generate)
    out = tmp_path / "run"
    args = [
        "--out",
        str(out),
        "--train",
        "2",
        "--calibration",
        "2",
        "--evaluation",
        "2",
        "--budgets",
        "2",
        "--context-len",
        "1",
        "--max-new-tokens",
        "8",
    ]
    assert evaluate.main([*args, "--collect-only"]) == 0
    assert not (out / "results.json").exists()
    assert evaluate.main([*args, "--resume"]) == 0
    before = json.loads((out / "results.json").read_text())
    assert evaluate.main(["--score-only", "--out", str(out)]) == 0
    after = json.loads((out / "results.json").read_text())
    assert before["estimators"] == after["estimators"]
    assert before["reference"] == after["reference"]
    assert calls == [1]
    with pytest.raises(ValueError, match="identical"):
        evaluate.main([*args, "--resume", "--target-fpr", "0.1"])
    with pytest.raises(SystemExit):
        evaluate.main(["--score-only", "--out", str(out), "--target-fpr", "0.1"])
    (out / "corpus.jsonl").write_text("corrupted\n")
    with pytest.raises(ValueError, match="checksum"):
        evaluate.main(["--score-only", "--out", str(out)])


def test_prompt_preparation_excludes_context_and_normalized_duplicates():
    import prepare_dolly

    prompt = "Explain how a neighborhood can organize a community garden."
    rows = [
        {"instruction": prompt, "category": "general_qa", "context": ""},
        {"instruction": prompt.upper(), "category": "general_qa", "context": ""},
        {"instruction": prompt, "category": "general_qa", "context": "reference passage"},
        {"instruction": "Too short", "category": "general_qa", "context": ""},
    ]
    selected, excluded = prepare_dolly.prepare(rows)
    assert selected == [{"text": prompt, "category": "general_qa", "source_row": 0}]
    assert excluded == {"normalized_duplicate": 1, "category_or_context": 1, "length": 1}
