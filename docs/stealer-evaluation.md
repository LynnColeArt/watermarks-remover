# Controlled evaluation of the stealer

The stealer estimates a **corpus contrast**, which is not automatically a watermark.
This harness asks whether it learns watermark preferences on unseen prompts when
model, tokenizer, and generation settings are controlled. It uses a local
Transformers SynthID implementation with public experimental keys. It makes no
claim about a vendor's model, keys, or detector.

## Run the pilot

The core scorer and metric tests need no ML packages. Generation is optional:

```sh
uv venv .venv-stealer
uv pip install --python .venv-stealer/bin/python -r stealer/requirements-evaluation.txt
.venv-stealer/bin/python stealer/evaluate.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --device cuda --train 64 --calibration 32 --evaluation 32 \
  --budgets 16,64 --context-len 4 --min-matches 5 --target-fpr 0.05 \
  --max-new-tokens 128 --batch-size 8 --seed 20260905 \
  --out work/stealer-pilot
```

Use `--device cpu` without a GPU. The model is downloaded from Hugging Face;
all generation runs locally, without a model API or API key. The pinned model
revision, library versions, hardware, settings, source hashes and elapsed time
are recorded. Exact outputs can vary across hardware/library versions. The
output directory must be new, so a rerun cannot silently mix or overwrite data.
Completed batches are saved atomically and can be resumed after interruption.
See the longer-run workflow below.

The bundled 256 prompts combine 16 topics with 16 tasks. A seeded shuffle assigns
unique prompt strings to disjoint training, calibration, and evaluation sets.
Topics and templates are shared across splits: this is an engineering pilot,
not evidence of broad topic generalization. Supply a larger, deduplicated JSONL
corpus (`{"text": "prompt"}`) using `--prompts` for larger studies. Exact duplicate
prompts are rejected; near-duplicate or semantic overlap is not detected.

## Experimental design

For every prompt the same model produces three independently seeded responses:

| Arm | Watermark | Purpose |
| --- | --- | --- |
| watermarked | On | Signal under test |
| baseline | Off | Reference distribution and calibration negatives |
| control | Off | Independent negative control |

Temperature, sampling limits, maximum length, prompt formatting, and batch size
are identical between arms. EOS ends a response; only generated tokens before
EOS are counted. Actual model token IDs are retained without a decode/re-encode
round trip. The estimator is given the correct context length (4 preceding
tokens for an n-gram length of 5). This is more information than an unknown
production deployment would necessarily provide.

At each nested training budget, one estimator contrasts watermarked versus
baseline responses and another contrasts control versus baseline responses.
Only **training token IDs** enter fitting. Neither keys nor g-values enter the
estimator. The second estimator tests whether sampling variation alone produces
apparent detection. A different baseline model is a later experiment, not the
control in this pilot.

The independent reference is a **keyed mean-g detector**: Transformers computes
g-values with the exact experimental generation keys, repeated contexts are
masked, and the remaining g-values are averaged across tokens and layers. This
is not a trained Bayesian detector. Its sampling table is constructed on the
same device as generation: CPU and CUDA RNGs produce different tables for the
same seed. The table hash is recorded, and optional CPU/CUDA positive-control
tests check real SynthID scoring without downloading a model. It is calibrated on the baseline calibration
split independently of each learned estimator. EOS and the first context-length
tokens are excluded. The original count estimator does not perform repeated-
context masking; that limitation is retained and reported.

A predeclared reference sanity check requires held-out AUROC at least 0.8 and
sensitivity at least 0.5. Failure means the generation/detection setup has not
established a usable watermark signal; do not interpret estimator results as
watermark recovery in that case. Passing is only a pilot sanity check.

## Metrics and interpretation

- **Coverage:** matched (context, token) pairs divided by eligible pairs on
  held-out text. The estimator requires at least `--min-matches` to score a
  document. This parameter must be chosen before inspecting evaluation results.
- **Calibration:** the threshold is the `ceil((n+1)*(1-target_fpr))`-th smallest
  valid baseline calibration score. Decisions require a score **strictly above**
  it, preserving conservative behavior on ties. If that rank exceeds `n`, the
  detector abstains: there is not enough scorable calibration data. This targets
  a conditional false-positive rate under exchangeability, not a guarantee of
  the observed test false-positive rate.
- **Detection and controls:** positive rates among decisions, Wilson 95%
  intervals, abstention counts, and detected fractions across all documents.
  An abstention never means watermark absent. With 32 negatives, even zero false
  positives has a Wilson upper bound of about 10.7%; the pilot cannot substantiate
  a production-level low false-positive rate.
- **Scorable AUROC:** watermark versus baseline separation among scorable
  held-out documents, computed without choosing a threshold. It can look good
  while coverage is poor; always read both.
- **Token alignment:** Spearman correlation between learned scores and known
  mean g-values on unique matched held-out pairs, excluding masked repeats.
  It is exploratory, conditional on the lookup table's selected tokens, and
  cannot establish recovery on unseen contexts or the entire vocabulary.

Budgets are nested and evaluated side by side. Do not choose a budget, threshold,
context length, or coverage cutoff using test results and then report that same
set as fresh validation. A wider study needs independent seeds, a larger prompt
corpus, more calibration negatives, and fresh evaluation prompts.

## Outputs

`report.md` is the readable summary. `results.json` retains configuration,
reference and estimator metrics, per-document scores, source hashes, and corpus
hash. `corpus.jsonl` contains prompts, response text, exact token IDs, generation
seeds, and keyed evaluation g-values. These are generated research data; keep
large runs outside the source tree or in a separately published dataset.

## Removal experiment gate

This harness does not implement or claim successful watermark removal. First
establish useful held-out coverage, token alignment, discrimination, and stable
false-positive behavior relative to the unwatermarked control. If that gate is
met, compare ordinary paraphrasing against scorer-assisted paraphrasing using the
independent keyed detector, matched sampling budgets, and meaning-preservation
assessment. A fall in the learned estimator's own score is not proof of removal.
If the gate fails, report that outcome before scaling data collection or building
a production rewrite integration.

## Sources

- [ETH SRI: Probing SynthID-Text](https://www.sri.inf.ethz.ch/blog/probingsynthid)
- [Google: SynthID Text documentation](https://ai.google.dev/responsible/docs/safeguards/synthid)
- [Transformers SynthID implementation, v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/logits_process.py)


## Longer runs and DGX Spark

`--resume` continues the same protocol and prompt split. Repeat the original
arguments with that flag; changing the model, runtime, dtype, device, source
files, batch size, budgets, or split is rejected. Each completed batch/arm is a
checksummed JSON shard. A process lock prevents simultaneous writers and is
released by the OS when a process exits or is killed. A partial temporary write
is ignored; a damaged committed shard raises an error. Existing complete runs
are verified by corpus hash and can be rescored without loading the model.

```sh
# Prepare pinned, attributed instructions (no reference answers are retained).
python stealer/prepare_dolly.py --out work/dolly

# Collect a larger corpus, without starting statistical analysis yet.
python stealer/evaluate.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --device cuda --dtype float32 --prompts work/dolly/prompts.jsonl \
  --train 1024 --calibration 256 --evaluation 256 --budgets 64,256,1024 \
  --context-len 4 --min-matches 5 --target-fpr 0.05 \
  --max-new-tokens 128 --batch-size 32 --seed 20260906 \
  --out work/dolly-scale --collect-only

# After interruption: repeat that exact collection command with --resume.
# After completion: score on any machine with the stdlib-only core.
python stealer/evaluate.py --score-only --out work/dolly-scale
```

The preparer retains context-free instructions in `open_qa`, `general_qa`,
`brainstorming`, and `creative_writing`, requires 40-1000 characters, and removes
NFKC/case/whitespace-normalized duplicates. It does not truncate instructions or
use their human answers. Semantic overlap remains possible. The resulting
prompt data retain Dolly's CC BY-SA 3.0 license and include attribution,
filtering counts, dataset revision, and source/output hashes. If needed,
`--source-file` accepts an offline copy only if its hash matches the pinned data.

This larger protocol changes the corpus and hardware relative to the initial
pilot. Compare its nested budgets *within this run* to isolate data volume;
do not attribute cross-run differences solely to more data. The initial scale
study still uses a small model and one seed, not a production watermark.

On Spark, use a CUDA-enabled ARM64/GB10 PyTorch runtime. Do not replace the
platform PyTorch with the workstation's wheel. In a GPU-enabled container,
create a project-local venv with `--system-site-packages` to inherit working
PyTorch, then install only `transformers==4.57.6` and `pytest==9.1.1` into it.
The inherited container's unrelated vLLM packages may have incompatible
Transformers requirements; they are not part of this research process. The
base container remains unchanged. Run the focused tests inside that environment
before collection:

```sh
python -m pytest tests/test_stealer.py tests/test_stealer_evaluation.py \
  tests/test_stealer_checkpoints.py -o addopts= -q
```

Generation metadata records the actual Torch/CUDA versions, tokenizer and
sampling-table hashes, dtype, device, dependency versions, and source hashes.
`manifest.json` freezes the protocol and split, `runtime.json` freezes the
runtime, and `generation.json` marks a complete corpus. `--score-only` accepts
only `--out` and uses the saved protocol; it cannot silently retune thresholds
on the held-out set. Analysis source hashes are recorded separately.
