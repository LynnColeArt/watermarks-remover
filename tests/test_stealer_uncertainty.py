"""Evidence symmetry, uncertainty behavior, and missing-context isolation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stealer"))
import uncertainty
from scorer import context_key, score_sequence
from tokens import count_ngrams


def test_swap_arms_reverses_evidence_without_changing_uncertainty():
    a = uncertainty.contrast(20, 100, 5, 100, 10)
    b = uncertainty.contrast(5, 100, 20, 100, 10)
    assert a["score"] == pytest.approx(-b["score"])
    assert a["standard_error"] == pytest.approx(b["standard_error"])
    assert uncertainty.contrast(5, 100, 5, 100, 10)["score"] == 0


def test_more_observations_increase_reliability():
    small = uncertainty.contrast(2, 10, 1, 10, 2)
    large = uncertainty.contrast(200, 1000, 100, 1000, 2)
    assert 0 < small["reliability"] < large["reliability"] < 1
    assert small["standard_error"] > large["standard_error"]
    assert abs(small["score"]) < abs(small["log_ratio"])


def test_baseline_only_tokens_are_negative_and_missing_contexts_abstain():
    wm = count_ngrams(["a x", "a x", "z x"], 1)
    base = count_ngrams(["a y", "a y", "q y"], 1)
    table = uncertainty.build_scorer(wm, base, 1)
    entries = {r["token"]: r for r in table["scorer"][context_key(("a",))]}
    assert entries["x"]["score"] > 0 > entries["y"]["score"]
    assert score_sequence(table, ["z", "x"], 1)["mean"] is None
    assert uncertainty.contrast(1, 1, 0, 0, 2) is None


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        uncertainty.contrast(2, 1, 0, 1, 2)
    with pytest.raises(ValueError):
        uncertainty.contrast(0, 1, 0, 1, 2, alpha=0)


def test_candidate_evaluation_preserves_abstention_and_effect_sign():
    from prefix_experiment import candidate_metrics

    table = {
        "scorer": {
            context_key(("1", "2", "3", "4")): [
                {"token": "5", "score": -1.0},
                {"token": "6", "score": 1.0},
            ]
        }
    }
    groups = [
        {
            "context": [1, 2, 3, 4],
            "candidates": [
                {"token": 5, "p_base": 0.2, "log_ratio": -0.5},
                {"token": 6, "p_base": 0.3, "log_ratio": 0.5},
                {"token": 7, "p_base": 0.5, "log_ratio": -1.0},
            ],
        }
    ]
    result = candidate_metrics(table, groups)
    assert result["pair_coverage"] == pytest.approx(2 / 3)
    assert result["baseline_mass_coverage_within_top_candidates"] == 0.5
    assert result["sign_accuracy_for_absolute_effect_at_least_0.2"] == 1.0
    groups[0]["context"] = [9, 9, 9, 9]
    result = candidate_metrics(table, groups)
    assert result["matched_pairs"] == 0
    assert result["spearman_on_matched_pairs"] is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_priming_matches_true_prefix_g_values_and_repetition_mask(device):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from prefix_experiment import prime_processor

    config = tf.SynthIDTextWatermarkingConfig(
        ngram_len=5,
        keys=[1, 2, 3, 4],
        skip_first_ngram_calls=True,
    )
    ids = torch.tensor([[9, 8, 1, 2, 3, 4]], device=device)
    scores = torch.arange(32, device=device, dtype=torch.float32).reshape(1, 32) / 32
    proc = prime_processor(config, 32, device, ids)
    actual = proc(ids, scores.clone())
    ngrams = torch.cat(
        (ids[:, -4:].expand(32, 4), torch.arange(32, device=device).reshape(32, 1)), 1
    )
    g = proc.compute_g_values(ngrams)[:, 0, :].unsqueeze(0)
    expected = proc.update_scores(scores, g)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(actual.softmax(-1), scores.softmax(-1))
    # The same suffix later in this generation must be skipped, not rewatermarked.
    for token in [1, 2, 3, 4]:
        ids = torch.cat((ids, torch.tensor([[token]], device=device)), 1)
        result = proc(ids, scores.clone())
    assert torch.equal(result, scores)


def test_cli_build_selects_new_estimator_and_keeps_legacy_default(tmp_path):
    import json

    import steal

    wm, base, out = (tmp_path / name for name in ("wm.jsonl", "base.jsonl", "s.json"))
    wm.write_text(json.dumps({"reply": "a x a x"}) + "\n")
    base.write_text(json.dumps({"reply": "a y a y"}) + "\n")
    args = ["build", "--replies", str(wm), "--baseline", str(base), "--ctx", "1", "--out", str(out)]
    assert steal.main([*args, "--estimator", "uncertainty"]) == 0
    data = json.loads(out.read_text())
    assert data["config"]["alpha"] == 0.5
    assert data["config"]["baseline_fallback"] == "abstain"
    assert steal.main(args) == 0
    assert json.loads(out.read_text())["config"]["alpha"] == 0.4
