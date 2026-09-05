# Stealer pilot: coverage prevents calibrated detection

This small local experiment found a clear watermark signal in the generated
text, but insufficient held-out coverage for the count estimator to make
calibrated detection decisions. It does **not** show that watermark stealing
cannot work at larger budgets or with a different estimator.

## Setup

- Source: `79e11f64fd7011b6f945953f16c1c388e9b7bf10`, clean working tree.
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- PyTorch 2.8.0+cu128; Transformers 4.57.6; Python 3.11.15;
  NVIDIA RTX 4070. Completed in 155.8 seconds.
- 64 training, 32 calibration, and 32 evaluation prompts; three independently
  seeded arms per prompt (384 total responses), up to 128 generated tokens each.
- Watermark: Transformers SynthID, nine public keys, n-gram length 5,
  repeated-context history 1024. Estimator receives the true tokenizer/context.
- Target FPR 5%, at least five matched pairs per document. These pilot settings
  were fixed before held-out results; thresholds used only calibration negatives.

## Observations

The keyed reference detected 32/32 watermarked test responses (Wilson 95% CI
89.3–100%), with 1/32 false positives on baseline outputs and 0/32 on the
independent unwatermarked control. Reference AUROC was 0.9883. These small counts
have wide uncertainty: the 0/32 false-positive result still has a 10.7% upper
Wilson bound. The reference passed the predeclared pilot sanity check.

| Training contrast | Prompts per arm | Watermarked test pair coverage | Scorable calibration negatives | Unique held-out matched pairs | Spearman score vs keyed g |
| --- | ---: | ---: | ---: | ---: | ---: |
| Watermarked vs baseline | 16 | 1.11% | 0/32 | 47 | -0.051 |
| Watermarked vs baseline | 64 | 2.49% | 1/32 | 127 | 0.083 |
| Unwatermarked control vs baseline | 16 | 0.41% | 0/32 | 38 | 0.257 |
| Unwatermarked control vs baseline | 64 | 1.64% | 2/32 | 114 | -0.014 |

At this calibration rule and FPR target, at least 19 scorable calibration
negatives are needed. All learned estimators therefore abstained on all test
arms. That is **missing evidence**, not zero sensitivity or successful removal.
Token correlations are descriptive, conditional on selected matched pairs; they
are not statistically confirmed recovery of the watermark rules.

The small model, short responses, single seed, and shared prompt templates limit
generalization. This is a useful test of the research pipeline, not a replacement
for a larger study. The result supports addressing exact-context coverage and
collecting enough calibration data before adding scorer-assisted paraphrasing.
No removal comparison was run because the estimator validation gate was unmet.

## Reproduce and inspect

Follow [the methodology and setup](../../docs/stealer-evaluation.md), then run:

```sh
python stealer/evaluate.py --model HuggingFaceTB/SmolLM2-135M-Instruct --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --device cuda --train 64 --calibration 32 --evaluation 32 --budgets 16,64 --context-len 4 --min-matches 5 --target-fpr 0.05 --max-new-tokens 128 --batch-size 8 --seed 20260905 --out work/stealer-pilot
```

[Machine-readable summary](results.summary.json) contains every aggregate,
calibration threshold/count, configuration, source hash, and the raw corpus
SHA-256. Large generated/per-document data are omitted from this source-tree
artifact; a rerun writes `corpus.jsonl`, `results.json`, and `report.md` together.
The runtime sampling-table hash is recorded because CPU and CUDA RNGs differ
even for identical SynthID keys and seeds.
