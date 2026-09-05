# DGX Spark scaling experiment

[Protocol](protocol.json) was committed before collecting or inspecting the
held-out results. It fixes one local model and compares nested training budgets
of 64, 256, and 1,024 prompts per arm on a new, deduplicated Dolly corpus.
Calibration and evaluation each use 256 prompts; all three arms generate at
most 128 tokens per response. This is 4,608 responses in total.

The model is still SmolLM2-135M-Instruct. The goal is to isolate data-volume
effects within a run before adding larger-model and cross-model confounders.
This remains a local SynthID experiment with known experimental keys, not a
vendor detector. Source and prompt provenance are in the protocol and runtime
artifacts. Prompt data retain their CC BY-SA 3.0 attribution.

## Operational validation

- Full local suite: 1,245 passed; 16 optional skips.
- Focused Spark suite: 30 passed, including real CPU/CUDA positive controls.
- A real-model run was deliberately interrupted immediately after the first
  batch was committed. Resuming and running an independent uninterrupted copy
  produced exactly the same corpus SHA-256:
  `cb808c57869e9c67737a19b95a5414f4ccf4b744d70b6a98754ad13aee680606`.
- The resumed corpus was also scored through the separate offline scoring path.

See [the longer-run workflow](../../docs/stealer-evaluation.md#longer-runs-and-dgx-spark)
for setup, collection, resumption, and scoring. The generation runtime uses the
Spark's CUDA-enabled ARM64 PyTorch, with Transformers 4.57.6 in an isolated
project environment. It does not replace the host or container's shared runtime.
