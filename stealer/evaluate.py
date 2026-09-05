#!/usr/bin/env python3
"""Controlled, local SynthID estimator evaluation; no model API calls.

Heavy dependencies are imported only by generation. Metrics and tests use stdlib.
The keyed reference detector is mean g-value with repetition masking, calibrated
on held-out unwatermarked samples. It is not a vendor or pretrained detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

import scorer
from checkpoints import Checkpoints, atomic_json, file_digest, writer_lock
from tokens import count_ngrams

# Public experimental keys, never a production watermark configuration.
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
TOPICS = [
    "a community garden",
    "a lighthouse",
    "a bicycle workshop",
    "a public library",
    "a mountain trail",
    "a neighborhood bakery",
    "a railway station",
    "a riverside park",
    "an astronomy club",
    "a pottery class",
    "a wildlife shelter",
    "a local history museum",
    "a farmers market",
    "a school orchestra",
    "a small theater",
    "a coastal village",
]
TASKS = [
    "Describe a visitor's first afternoon at {topic}.",
    "Write a short story about two friends helping at {topic}.",
    "Explain how volunteers could improve {topic}.",
    "Describe the sounds, colors, and textures around {topic}.",
    "Write a friendly introduction to {topic} for a newcomer.",
    "Tell a story about an unexpected discovery at {topic}.",
    "Describe preparations for a celebration at {topic}.",
    "Explain what a child might learn from a visit to {topic}.",
    "Write about a rainy morning at {topic}.",
    "Describe how {topic} changes with the seasons.",
    "Imagine a conversation between two people meeting at {topic}.",
    "Tell a story about solving a practical problem at {topic}.",
    "Describe a quiet evening at {topic}.",
    "Write a letter thanking the people who care for {topic}.",
    "Explain how to plan a welcoming event at {topic}.",
    "Describe a memorable act of kindness at {topic}.",
]


def split_prompts(prompts, train, calibration, evaluation, seed):
    """Reject duplicate prompts before a seeded, disjoint split."""
    if min(train, calibration, evaluation) < 1:
        raise ValueError("all split sizes must be positive")
    if len(set(prompts)) != len(prompts):
        raise ValueError("duplicate prompts would contaminate the split")
    if len(prompts) < train + calibration + evaluation:
        raise ValueError("not enough unique prompts for the requested splits")
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - reproducible research split
    return {
        "train": shuffled[:train],
        "calibration": shuffled[train : train + calibration],
        "evaluation": shuffled[train + calibration : train + calibration + evaluation],
    }


def calibrate(scores, fpr):
    """Conservative order statistic; decisions use strictly greater than.

    This finite-sample rank requires enough *scorable* calibration documents.
    It is a conditional, exchangeability-based target, not a measured guarantee.
    """
    if not 0 < fpr < 1:
        raise ValueError("target FPR must be between zero and one")
    values = sorted(x for x in scores if x is not None)
    rank = math.ceil((len(values) + 1) * (1 - fpr))
    if not values or rank > len(values):
        return {"status": "insufficient_calibration", "threshold": None, "n": len(values)}
    return {"status": "calibrated", "threshold": values[rank - 1], "n": len(values)}


def wilson(positive, total):
    if not total:
        return None
    z = 1.959963984540054
    p = positive / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(scores, calibration):
    threshold = calibration["threshold"]
    valid = [x for x in scores if x is not None] if threshold is not None else []
    positive = sum(x > threshold for x in valid)
    return {
        "documents": len(scores),
        "decisions": len(valid),
        "abstentions": len(scores) - len(valid),
        "positive": positive,
        "positive_rate_among_decisions": positive / len(valid) if valid else None,
        "positive_rate_95ci": wilson(positive, len(valid)),
        "detected_fraction_all_documents": positive / len(scores) if valid else None,
    }


def auc(positive, negative):
    """Pairwise AUROC on scorable documents only; ties receive half credit."""
    positive = [x for x in positive if x is not None]
    negative = [x for x in negative if x is not None]
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def rank_correlation(pairs):
    """Spearman correlation with average ranks for ties; undefined -> null."""
    if len(pairs) < 3:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            for i in order[start:end]:
                result[i] = (start + end - 1) / 2
            start = end
        return result

    x, y = (ranks([p[i] for p in pairs]) for i in (0, 1))
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def fit(rows, arm, context_len, budget):
    """The estimator sees only training token IDs, never keys or g-values."""
    training = [row for row in rows if row["split"] == "train"]

    def counts(which):
        selected = islice((row for row in training if row["arm"] == which), budget)
        sequences = ([str(t) for t in row["token_ids"]] for row in selected)
        return count_ngrams(sequences, context_len, lambda sequence: sequence)

    return scorer.build_scorer(counts(arm), counts("baseline"), context_len)


def evaluate_table(table, rows, context_len, min_matches, target_fpr):
    scored = []
    alignment = {}
    lookup = {
        ctx: {x["token"]: x["score"] for x in items} for ctx, items in table["scorer"].items()
    }
    for row in rows:
        if row["split"] == "train":
            continue
        tokens = [str(t) for t in row["token_ids"]]
        result = scorer.score_sequence(table, tokens, context_len)
        value = result["mean"] if result["applied"] >= min_matches else None
        scored.append(
            {"id": row["id"], "split": row["split"], "arm": row["arm"], "value": value, **result}
        )
        # Use each held-out (context, token) once; exclude repeated contexts.
        # g-values are used only for evaluation, after fitting the table.
        if row["split"] == "evaluation":
            for i in range(context_len, len(tokens)):
                g = row["g_values"][i - context_len]
                ctx = scorer.context_key(tokens[i - context_len : i])
                score = lookup.get(ctx, {}).get(tokens[i])
                if g is not None and score is not None:
                    alignment[(ctx, tokens[i])] = (score, g)

    calibration = calibrate(
        [
            row["value"]
            for row in scored
            if row["split"] == "calibration" and row["arm"] == "baseline"
        ],
        target_fpr,
    )
    arms = {}
    values = {}
    for arm in ("watermarked", "baseline", "control"):
        selected = [row for row in scored if row["split"] == "evaluation" and row["arm"] == arm]
        values[arm] = [row["value"] for row in selected]
        arms[arm] = summarize(values[arm], calibration)
        arms[arm]["pair_coverage"] = sum(r["applied"] for r in selected) / max(
            1, sum(r["eligible"] for r in selected)
        )
    return {
        "calibration": calibration,
        "arms": arms,
        "scorable_auroc": auc(values["watermarked"], values["baseline"]),
        "heldout_unique_matched_pairs": len(alignment),
        "heldout_spearman_s_vs_g": rank_correlation(list(alignment.values())),
        "document_scores": scored,
    }


def reference_results(rows, target_fpr):
    calibration = calibrate(
        [
            row["reference_score"]
            for row in rows
            if row["split"] == "calibration" and row["arm"] == "baseline"
        ],
        target_fpr,
    )
    values = {
        arm: [
            row["reference_score"]
            for row in rows
            if row["split"] == "evaluation" and row["arm"] == arm
        ]
        for arm in ("watermarked", "baseline", "control")
    }
    arms = {arm: summarize(scores, calibration) for arm, scores in values.items()}
    reference_auc = auc(values["watermarked"], values["baseline"])
    sensitivity = arms["watermarked"]["positive_rate_among_decisions"]
    # Predeclared pilot sanity check, not a production validation criterion.
    passed = (
        reference_auc is not None
        and reference_auc >= 0.8
        and sensitivity is not None
        and sensitivity >= 0.5
    )
    return {
        "name": "keyed mean-g detector, repetition-masked, empirical calibration",
        "calibration": calibration,
        "arms": arms,
        "auroc": reference_auc,
        "sanity_check": "passed" if passed else "failed",
        "sanity_rule": "held-out reference AUROC >= 0.8 and sensitivity >= 0.5",
    }


def make_reference(config, vocab_size, device):
    """SynthID tables depend on the RNG device, not just the seed and keys."""
    return config.construct_processor(vocab_size, device)


def keyed_scores(sequence, reference, context_len):
    """Reference g-values on the generation device, masking repeated contexts."""
    import torch

    if len(sequence) <= context_len:
        return [], None
    ids = torch.tensor([sequence], dtype=torch.long, device=reference.device)
    g = reference.compute_g_values(ids).float().mean(dim=-1)[0].tolist()
    mask = reference.compute_context_repetition_mask(ids)[0].tolist()
    gs = [value if keep else None for value, keep in zip(g, mask, strict=True)]
    valid = [x for x in gs if x is not None]
    return gs, statistics.mean(valid) if valid else None


def generate(args, splits, out, checkpoint=None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    fingerprint = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=False,
            torch_dtype=getattr(torch, getattr(args, "dtype", "float32")),
            use_safetensors=True,
        )
        .to(args.device)
        .eval()
    )
    wm_config = SynthIDTextWatermarkingConfig(
        ngram_len=args.context_len + 1, keys=KEYS, skip_first_ngram_calls=True
    )
    reference = make_reference(wm_config, model.config.vocab_size, args.device)
    runtime = {
        "model": args.model,
        "resolved_model_revision": model.config._commit_hash,
        "tokenizer_sha256": fingerprint,
        "tokenizer_kind": "model token IDs, no decode/re-encode in estimator",
        "watermark": wm_config.to_dict(),
        "sampling_table_sha256": hashlib.sha256(
            reference.sampling_table.cpu().numpy().tobytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
    }
    from importlib.metadata import version

    runtime["dependencies"] = {
        name: version(name) for name in ("numpy", "tokenizers", "huggingface_hub", "safetensors")
    }
    runtime["cuda_runtime"] = torch.version.cuda
    runtime["dtype"] = str(model.dtype)
    if checkpoint:
        checkpoint.bind_runtime(runtime)
    rows = []
    with (out / ".corpus.jsonl.tmp").open("w", encoding="utf-8") as fh:
        for split_index, (split, prompts) in enumerate(splits.items()):
            for start in range(0, len(prompts), args.batch_size):
                batch = prompts[start : start + args.batch_size]
                formatted = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                    )
                    if tokenizer.chat_template
                    else p
                    for p in batch
                ]
                inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(args.device)
                for arm_index, arm in enumerate(("watermarked", "baseline", "control")):
                    # Independent streams; all three arms use identical generation settings.
                    seed = args.seed + split_index * 100000 + start * 3 + arm_index
                    cached = checkpoint.load(split, start, arm, batch, seed) if checkpoint else None
                    if cached is not None:
                        rows.extend(cached)
                        for row in cached:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        continue
                    transformers.set_seed(seed)
                    with torch.inference_mode():
                        output = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.95,
                            top_k=50,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            watermarking_config=wm_config if arm == "watermarked" else None,
                        )
                    batch_rows = []
                    for offset, generated_ids in enumerate(
                        output[:, inputs.input_ids.shape[1] :].tolist()
                    ):
                        sequence = generated_ids
                        if tokenizer.eos_token_id in sequence:
                            sequence = sequence[: sequence.index(tokenizer.eos_token_id)]
                        gs, reference_score = keyed_scores(sequence, reference, args.context_len)
                        row = {
                            "id": f"{split}-{start + offset}",
                            "split": split,
                            "arm": arm,
                            "seed": seed,
                            "prompt": batch[offset],
                            "token_ids": sequence,
                            "text": tokenizer.decode(sequence, skip_special_tokens=True),
                            "g_values": gs,
                            "reference_score": reference_score,
                        }
                        batch_rows.append(row)
                    if checkpoint:
                        checkpoint.save(batch_rows, split, start, arm, batch, seed)
                    rows.extend(batch_rows)
                    for row in batch_rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                print(
                    f"{split}: {min(start + args.batch_size, len(prompts))}/{len(prompts)} prompts, three arms",
                    flush=True,
                )
    os.replace(out / ".corpus.jsonl.tmp", out / "corpus.jsonl")
    return rows, runtime


def report(result):
    def rate(arm):
        value = arm["positive_rate_among_decisions"]
        if value is None:
            return f"n/a ({arm['abstentions']} abstain)"
        lo, hi = arm["positive_rate_95ci"]
        return f"{value:.1%} [{lo:.1%}, {hi:.1%}]; {arm['abstentions']} abstain"

    lines = [
        "# Controlled stealer evaluation",
        "",
        "Local experimental SynthID configuration; no vendor-watermark claims.",
        "",
        f"Reference sanity check: **{result['reference']['sanity_check']}** "
        f"({result['reference']['sanity_rule']}). A failed check prevents interpreting "
        "estimator performance as a watermark-recovery result.",
        "",
        "Positive rates below are conditional on a decision; brackets are Wilson 95% intervals. "
        "Abstentions are not negative detections. Unwatermarked arms measure false positives.",
        "",
        "| Estimator | Training prompts/arm | Watermarked positive rate | Baseline false positives | Independent control false positives | WM pair coverage |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ]
    ref = result["reference"]["arms"]
    lines.append(
        f"| Keyed reference | — | {rate(ref['watermarked'])} | {rate(ref['baseline'])} | {rate(ref['control'])} | — |"
    )
    for run in result["estimators"]:
        arms = run["arms"]
        lines.append(
            f"| {run['training_arm']} vs baseline | {run['budget']} | {rate(arms['watermarked'])} | {rate(arms['baseline'])} | {rate(arms['control'])} | {arms['watermarked']['pair_coverage']:.2%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The control-trained estimator sees two independent unwatermarked samples from the same model. "
        "Any apparent watermark discrimination there is a negative-control result, not watermark recovery.",
        "",
        (
            "The corpus uses disjoint prompts from an external corpus. Check its provenance and deduplication policy; "
            "disjoint strings do not guarantee disjoint topics or meanings. "
            if result["config"].get("prompts")
            else "The corpus uses disjoint prompts drawn from shared templates and topics. This is an engineering pilot, "
            "not a representative language benchmark. "
        )
        + "Generation seeds are independent across arms. Training budgets are nested and must not be selected using evaluation results.",
        "",
        "The keyed reference uses the actual generation keys and the Transformers g-function, "
        "with repeated contexts masked. It is a mean-g detector calibrated on separate unwatermarked outputs, "
        "not Google's private detector or a trained Bayesian detector. Keys/g-values never enter estimator fitting.",
        "",
        "Calibration targets a conditional false-positive rate under exchangeability. Small samples give broad intervals; "
        "scorable AUROC and token-rank correlations are exploratory and conditional on coverage. "
        "Repeated-context masking is applied to the reference/alignment, while the original count estimator remains unchanged.",
        "",
        "No removal experiment is included: validate held-out estimation before interpreting scorer-assisted rewrites. "
        "A future removal comparison must use this independent keyed detector and meaning-preservation assessment, "
        "and retain ordinary paraphrasing as a baseline.",
        "",
        "## Reproduce",
        "",
        "```sh",
        result["command"],
        "```",
        "",
        "Full configuration, calibration sizes, per-document scores, token alignment, and AUROC: `results.json`. "
        "Generated text, IDs, masks, and prompts: `corpus.jsonl`. Runtime/model versions: `metadata` in results.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--resume", action="store_true", help="resume identical planned batches")
    p.add_argument(
        "--collect-only", action="store_true", help="generate/checkpoint without fitting"
    )
    p.add_argument(
        "--score-only",
        action="store_true",
        help="score a complete saved run without ML dependencies",
    )
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--calibration", type=int, default=32)
    p.add_argument("--evaluation", type=int, default=32)
    p.add_argument("--budgets", default="16,64")
    p.add_argument("--context-len", type=int, default=4)
    p.add_argument("--min-matches", type=int, default=5)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument(
        "--prompts", type=Path, help="optional JSONL corpus, text field; unique prompts required"
    )
    p.add_argument(
        "--out", type=Path, required=True, help="new output directory; existing paths rejected"
    )
    args = p.parse_args(argv)
    if args.score_only:
        supplied = sys.argv[1:] if argv is None else argv
        if any(
            x.startswith("--") and x.split("=")[0] not in ("--score-only", "--out")
            for x in supplied
        ):
            p.error("--score-only uses the saved protocol; supply only --score-only and --out")
        manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
        for key, value in manifest["config"].items():
            setattr(args, key, Path(value) if key == "prompts" and value else value)
    budgets = sorted(set(int(x) for x in args.budgets.split(",")))
    if not budgets or min(budgets) < 1 or max(budgets) > args.train:
        p.error("budgets must be positive and no greater than --train")
    if (
        min(args.context_len, args.min_matches, args.batch_size) < 1
        or args.max_new_tokens <= args.context_len
    ):
        p.error("positive context/match/batch sizes and a longer generation are required")
    if not 0 < args.target_fpr < 1:
        p.error("target FPR must be between zero and one")
    if args.score_only:
        splits = manifest["splits"]
    else:
        prompts = (
            [
                json.loads(line)["text"]
                for line in args.prompts.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if args.prompts
            else [task.format(topic=topic) for task in TASKS for topic in TOPICS]
        )
        splits = split_prompts(prompts, args.train, args.calibration, args.evaluation, args.seed)
        config = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("out", "resume", "collect_only", "score_only")
        }
        manifest = {
            "schema": 1,
            "config": config,
            "splits": splits,
            "source_sha256": source_hashes(),
        }
    if args.resume or args.score_only:
        if not args.out.is_dir():
            p.error("resume/scoring requires an existing experiment directory")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    with writer_lock(args.out):
        checkpoint = Checkpoints(args.out, manifest, resume=args.resume or args.score_only)
        start = time.monotonic()
        completed = checkpoint.completed()
        if completed:
            rows, metadata = completed
        elif args.score_only:
            p.error("generation is incomplete; resume collection before scoring")
        else:
            rows, metadata = generate(args, splits, args.out, checkpoint=checkpoint)
            metadata["generation_seconds_this_session"] = time.monotonic() - start
            metadata["source_sha256"] = manifest["source_sha256"]
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                cwd=Path(__file__).parent,
                text=True,
            ).strip()
            metadata["working_tree_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],  # noqa: S607
                    cwd=Path(__file__).parent,
                    text=True,
                ).strip()
            )
            checkpoint.finish(rows, metadata)
        if args.collect_only:
            print(f"collection complete: {len(rows)} responses in {args.out}", flush=True)
            return 0
        return write_results(args, splits, budgets, rows, metadata, argv)


def source_hashes():
    return {path.name: file_digest(path) for path in Path(__file__).parent.glob("*.py")}


def write_results(args, splits, budgets, rows, metadata, argv=None):
    metadata = dict(metadata)
    start = time.monotonic()
    estimators = []
    for budget in budgets:
        for arm in ("watermarked", "control"):
            table = fit(rows, arm, args.context_len, budget)
            evaluated = evaluate_table(
                table, rows, args.context_len, args.min_matches, args.target_fpr
            )
            estimators.append(
                {
                    "budget": budget,
                    "training_arm": arm,
                    "contexts": len(table["scorer"]),
                    **evaluated,
                }
            )
    metadata["analysis_source_sha256"] = source_hashes()
    metadata["analysis_seconds"] = time.monotonic() - start
    metadata["corpus_sha256"] = file_digest(args.out / "corpus.jsonl")
    import shlex

    generation_args = ["python", "stealer/evaluate.py"]
    for key, value in vars(args).items():
        if key in ("resume", "collect_only", "score_only") or value is None:
            continue
        recorded_value = (
            (metadata.get("resolved_model_revision") or value) if key == "revision" else value
        )
        generation_args.extend(["--" + key.replace("_", "-"), str(recorded_value)])
    command = shlex.join(generation_args)
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metadata": metadata,
        "command": command,
        "prompt_split_sha256": hashlib.sha256(
            json.dumps(splits, sort_keys=True).encode()
        ).hexdigest(),
        "reference": reference_results(rows, args.target_fpr),
        "estimators": estimators,
    }
    atomic_json(args.out / "results.json", result)
    (args.out / "report.md").write_text(report(result), encoding="utf-8")
    print(f"wrote {args.out / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
