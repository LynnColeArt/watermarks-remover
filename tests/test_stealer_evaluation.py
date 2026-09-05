"""Metrics and isolation checks; no model downloads or GPU required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stealer"))

import evaluate
import scorer
import steal
import tokens


def test_split_is_disjoint_deterministic_and_rejects_duplicates():
    prompts = [f"prompt {i}" for i in range(12)]
    splits = evaluate.split_prompts(prompts, 6, 3, 3, 42)
    assert splits == evaluate.split_prompts(prompts, 6, 3, 3, 42)
    assert len({prompt for group in splits.values() for prompt in group}) == 12
    with pytest.raises(ValueError, match="duplicate"):
        evaluate.split_prompts([*prompts, prompts[0]], 6, 3, 3, 42)
    with pytest.raises(ValueError, match="enough"):
        evaluate.split_prompts(prompts, 10, 3, 3, 42)


def test_calibration_requires_sample_size_and_preserves_ties():
    assert evaluate.calibrate([None] * 100, 0.05)["threshold"] is None
    assert evaluate.calibrate(list(range(18)), 0.05)["threshold"] is None
    calibrated = evaluate.calibrate(list(range(19)), 0.05)
    assert calibrated["threshold"] == 18
    result = evaluate.summarize([18, 19, None], calibrated)
    assert result["positive"] == 1
    assert result["decisions"] == 2
    assert result["abstentions"] == 1
    assert result["positive_rate_among_decisions"] == 0.5
    assert result["detected_fraction_all_documents"] == pytest.approx(1 / 3)
    ties = evaluate.calibrate([0.0] * 100, 0.05)
    assert evaluate.summarize([0.0] * 20, ties)["positive"] == 0


def test_failed_calibration_does_not_produce_negative_verdicts():
    result = evaluate.summarize([1.0, 2.0], {"threshold": None})
    assert result["decisions"] == 0
    assert result["abstentions"] == 2
    assert result["positive_rate_among_decisions"] is None
    assert result["positive_rate_95ci"] is None


def test_auc_and_uncertainty_have_known_results():
    assert evaluate.auc([2, 3], [0, 1]) == 1
    assert evaluate.auc([0, 1], [2, 3]) == 0
    assert evaluate.auc([1, 1], [1, 1]) == 0.5
    assert evaluate.auc([None], [0]) is None
    lo, hi = evaluate.wilson(0, 32)
    assert lo == pytest.approx(0)
    assert hi == pytest.approx(0.107179, abs=1e-6)
    assert evaluate.rank_correlation([(0, 3), (1, 2), (2, 1)]) == pytest.approx(-1)
    assert evaluate.rank_correlation([(0, 3), (0, 2), (0, 1)]) is None


def test_fitting_cannot_see_evaluation_data_or_reference_keys():
    rows = [
        {"split": "train", "arm": "watermarked", "token_ids": [1, 2, 1, 2]},
        {"split": "train", "arm": "baseline", "token_ids": [1, 3, 1, 3]},
        {"split": "evaluation", "arm": "watermarked", "token_ids": [9, 9, 9]},
    ]
    before = evaluate.fit(rows, "watermarked", 1, 1)
    rows[-1]["token_ids"] = [1000, 2000, 3000]
    rows[-1]["g_values"] = [1.0, 1.0]
    assert evaluate.fit(rows, "watermarked", 1, 1) == before
    assert '"9"' not in json.dumps(before)


def test_alignment_deduplicates_pairs_and_masks_repeats():
    counts = tokens.count_ngrams(["a b a b"], 1)
    table = scorer.build_scorer(counts, counts, 1)
    rows = [
        {
            "id": "heldout",
            "split": "evaluation",
            "arm": "watermarked",
            "token_ids": ["a", "b", "a", "b"],
            "g_values": [1.0, None, 1.0],
        }
    ]
    result = evaluate.evaluate_table(table, rows, 1, 1, 0.05)
    assert result["heldout_unique_matched_pairs"] == 1
    assert result["heldout_spearman_s_vs_g"] is None
    assert result["arms"]["watermarked"]["pair_coverage"] == 1
    assert result["calibration"]["status"] == "insufficient_calibration"


def test_model_tokenizer_preserves_ids_and_records_identity(monkeypatch):
    import types

    class FakeTokenizer:
        is_fast = True
        backend_tokenizer = types.SimpleNamespace(to_str=lambda: '{"vocab": "stable"}')

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [42] if text == "UPPER" else [43]

    auto = types.SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=auto))
    encode, config = tokens.load_tokenizer("test-model", "pinned-commit")
    assert encode("UPPER") == ["42"]
    assert encode("upper") == ["43"]
    assert config["revision"] == "pinned-commit"
    assert len(config["sha256"]) == 64


def test_detect_rejects_changed_tokenizer_identity(monkeypatch, tmp_path):
    config = {"kind": "huggingface", "name": "test", "revision": "abc", "sha256": "old"}
    path = tmp_path / "scorer.json"
    path.write_text(json.dumps({"config": {"tokenizer": config}, "scorer": {}}))
    monkeypatch.setattr(
        steal, "load_tokenizer", lambda *a: (lambda text: ["1"], {**config, "sha256": "new"})
    )
    assert steal.main(["detect", "--s-star", str(path), "--text", "example"]) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_synthid_positive_control_matches_generation_device(device):
    """No model download: known watermark must score above uniform draws.

    CUDA and CPU RNGs produce different tables for the same seed. This catches
    a CPU reference silently checking GPU generations with the wrong table.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = transformers.SynthIDTextWatermarkingConfig(
        ngram_len=3, keys=evaluate.KEYS, skip_first_ngram_calls=True
    )
    reference = evaluate.make_reference(config, 128, device)
    generator = config.construct_processor(128, device)
    torch.manual_seed(17)
    if device == "cuda":
        torch.cuda.manual_seed_all(17)
    sequences = torch.zeros((8, 1), device=device, dtype=torch.long)
    for _ in range(40):
        logits = generator(sequences, torch.zeros((8, 128), device=device))
        next_token = torch.multinomial(logits.softmax(dim=-1), 1)
        sequences = torch.cat([sequences, next_token], dim=1)
    values = [evaluate.keyed_scores(seq[1:], reference, 2)[1] for seq in sequences.tolist()]
    assert sum(values) / len(values) > 0.65
    assert torch.equal(reference.sampling_table, generator.sampling_table)
