# Repeated-prefix estimation experiment

This controlled local experiment tests whether concentrating samples on exact
prefixes improves recovery of watermark preferences at those prefixes. It also
measures the cost to coverage at unqueried contexts. This is a stronger access
setting than a text-only commercial API: the collector can reset watermark
history and condition on exact generated prefixes. Estimators receive only
sampled token IDs, never keys or model probabilities.

The public protocol was committed before collection. Both strategies receive
2,048 training tokens per arm and seed. Repeated sampling draws 64 independent
next tokens at each of 32 prefixes. Passive sampling draws a 64-token continuation
at each of those same prefixes. Three independent sampling/prompt seeds share
one fixed model and public watermark key. An unwatermarked control fit measures
spurious estimated watermark signal.

The new estimator keeps both positive and negative log-ratio estimates and
requires the context in both training corpora. It uses symmetric add-0.5
smoothing on the observed per-context union, a delta-method log-ratio variance,
and score = log-ratio / (1 + variance). The uncertainty is approximate, especially
at low counts. Shrinkage is a ranking heuristic, not a posterior probability or
certified confidence interval. Missing context or token remains unknown.

Evaluation uses the exact watermarked/unwatermarked log probability ratio on
the top 32 baseline next-token candidates. Scores at queried anchors measure
local recovery, not generalization. Results at 32 unqueried contexts are separate.
Candidate coverage and conditional ranking/sign accuracy must be read together.
The known-key oracle is used only after fitting; no detector FPR is claimed.
The previous document detector's calibration limitation remains unresolved.

HF SynthID initializes an empty context instead of reading the whole prompt.
The collector replays four skipped processor calls before sampling to load the
actual four-token suffix. A CPU/CUDA test compares its resulting distribution
against g-values computed directly for every candidate and checks repeat masking.
Special tokens are excluded before watermarking; all remaining vocabulary tokens
are sampled at temperature 1 without top-k/top-p truncation. Fixed-length
continuations provide equal generated-token budgets. Discovery (1,536 additional
unwatermarked tokens per seed) is shared and accounted separately. Query latency
and total inference work are not equal across strategies.

Run from the repository root with the optional evaluation dependencies:

```sh
python stealer/prefix_experiment.py --protocol benchmarks/stealer-prefix/protocol.json --out work/prefix-experiment --device cuda
python stealer/prefix_experiment.py --out work/prefix-experiment --score-only
```

Repeating collection with the same command reuses completed per-seed artifacts.
Source, protocol, device and runtime changes are rejected. Interruption within a
seed recomputes that entire seed. Score-only needs only Python's standard library
and records its analysis source hash. Raw experiment artifacts contain public
synthetic prompts and generated local-model data; no private Claude logs are used.
