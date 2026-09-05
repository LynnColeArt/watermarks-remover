#!/usr/bin/env python3
"""Local, equal-token-budget comparison of passive and repeated-prefix sampling.

The collector has controlled model access to reset SynthID history and condition
on exact prefixes. This is NOT an experiment against a commercial API. Per-seed
artifacts are atomic and resumable; --score-only needs no ML dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
from pathlib import Path

import evaluate
import scorer
import uncertainty
from checkpoints import atomic_json, file_digest, writer_lock
from tokens import count_ngrams


def training_counts(sequences, context_len=4):
    return count_ngrams(([str(t) for t in s] for s in sequences), context_len, lambda s: s)


def candidate_metrics(table, candidates):
    """Evaluate only after fitting. Missing estimates are unknown, not zero."""
    lookup = {
        ctx: {entry["token"]: entry["score"] for entry in entries}
        for ctx, entries in table["scorer"].items()
    }
    matched = []
    all_mass = matched_mass = 0.0
    within = []
    for group in candidates:
        ctx = scorer.context_key([str(t) for t in group["context"]])
        local = []
        for item in group["candidates"]:
            all_mass += item["p_base"]
            value = lookup.get(ctx, {}).get(str(item["token"]))
            if value is not None:
                matched_mass += item["p_base"]
                pair = (value, item["log_ratio"])
                matched.append(pair)
                local.append(pair)
        rho = evaluate.rank_correlation(local) if len(local) >= 3 else None
        if rho is not None:
            within.append(rho)
    nontrivial = [(s, y) for s, y in matched if abs(y) >= 0.2]
    n = sum(len(g["candidates"]) for g in candidates)
    return {
        "candidate_pairs": n,
        "matched_pairs": len(matched),
        "pair_coverage": len(matched) / n if n else 0,
        "baseline_mass_coverage_within_top_candidates": matched_mass / all_mass if all_mass else 0,
        "spearman_on_matched_pairs": evaluate.rank_correlation(matched),
        "mean_within_context_spearman": statistics.mean(within) if within else None,
        "contexts_with_rank_metric": len(within),
        "sign_accuracy_for_absolute_effect_at_least_0.2": (
            sum(s * y > 0 for s, y in nontrivial) / len(nontrivial) if nontrivial else None
        ),
        "nontrivial_matched_pairs": len(nontrivial),
    }


def analyze_seed(data):
    results = []
    for strategy, arms in data["training"].items():
        base = training_counts(arms["baseline"])
        for fit_arm in ("watermarked", "control"):
            counts = training_counts(arms[fit_arm])
            for name, builder in (
                ("legacy", scorer.build_scorer),
                ("uncertainty", uncertainty.build_scorer),
            ):
                table = builder(counts, base, 4)
                results.append(
                    {
                        "seed": data["seed"],
                        "strategy": strategy,
                        "fit_arm": fit_arm,
                        "estimator": name,
                        **{
                            split: candidate_metrics(table, groups)
                            for split, groups in data["oracle"].items()
                        },
                    }
                )
    return results


def prime_processor(config, vocab_size, device, prefixes):
    """Replay four skipped calls so the first scored token uses the true suffix.

    HF initializes its state with zeros rather than the supplied prompt. The
    replay appends the last four tokens across four transitions, including the
    first actual sampling call. No token is sampled during priming.
    """
    import torch

    proc = config.construct_processor(vocab_size, device)
    zero = torch.zeros((len(prefixes), vocab_size), device=device)
    for offset in range(4, 0, -1):
        proc(prefixes[:, :-offset], zero)
    return proc


def collect_seed(args, protocol, seed, model, tokenizer, config):
    import torch
    import transformers

    transformers.set_seed(seed)
    device = args.device
    topics = [task.format(topic=topic) for task in evaluate.TASKS for topic in evaluate.TOPICS]
    random.Random(seed).shuffle(topics)  # noqa: S311 - reproducible experimental sampling
    prompts = topics[: 2 * protocol["anchors"]]
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(device)
    banned = tokenizer.all_special_ids

    def logits_for(ids, mask):
        scores = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
        scores[:, banned] = -float("inf")
        return scores

    def continuation(ids, mask, steps, watermarked, sampling_seed):
        transformers.set_seed(sampling_seed)
        proc = (
            prime_processor(config, model.config.vocab_size, device, ids) if watermarked else None
        )
        original_len = ids.shape[1]
        for _ in range(steps):
            scores = logits_for(ids, mask)
            if proc is not None:
                scores = proc(ids, scores)
            token = torch.multinomial(scores.softmax(-1), 1)
            ids = torch.cat((ids, token), -1)
            mask = torch.cat((mask, torch.ones_like(token)), -1)
        return ids[:, original_len:]

    with torch.inference_mode():
        # Public, independent unwatermarked discovery; identical cost for both strategies.
        discovery = continuation(inputs.input_ids, inputs.attention_mask, 24, False, seed + 100)
        # A common cut at 24 preserves batching and keeps all four context tokens generated.
        prefixes = torch.cat((inputs.input_ids, discovery), -1)
        masks = torch.cat((inputs.attention_mask, torch.ones_like(discovery)), -1)
        contexts = prefixes[:, -4:].tolist()
        if len({tuple(c) for c in contexts}) != len(contexts):
            raise ValueError(
                "discovery produced duplicate contexts; protocol requires unique contexts"
            )
        base_logits = logits_for(prefixes, masks)
        proc = prime_processor(config, model.config.vocab_size, device, prefixes)
        wm_logits = proc(prefixes, base_logits.clone())
        p0, p1 = base_logits.softmax(-1), wm_logits.softmax(-1)
        top = p0.topk(protocol["candidates"], -1).indices
        oracle = {"anchor": [], "novel": []}
        for i, choices in enumerate(top.tolist()):
            records = [
                {
                    "token": token,
                    "p_base": float(p0[i, token]),
                    "p_watermarked": float(p1[i, token]),
                    "log_ratio": math.log(max(float(p1[i, token]), 1e-30))
                    - math.log(max(float(p0[i, token]), 1e-30)),
                }
                for token in choices
            ]
            oracle["anchor" if i < protocol["anchors"] else "novel"].append(
                {
                    "context": contexts[i],
                    "candidates": records,
                }
            )
        n = protocol["anchors"]
        training = {"repeated": {}, "passive": {}}
        for arm_index, arm in enumerate(("watermarked", "baseline", "control")):
            transformers.set_seed(seed + 1000 + arm_index)
            # Independent draws from the same fresh-history next-token distribution.
            distribution = p1[:n] if arm == "watermarked" else p0[:n]
            draws = torch.multinomial(distribution, protocol["draws"], replacement=True).tolist()
            training["repeated"][arm] = [
                contexts[i] + [token] for i, tokens in enumerate(draws) for token in tokens
            ]
            tails = continuation(
                prefixes[:n],
                masks[:n],
                protocol["draws"],
                arm == "watermarked",
                seed + 2000 + arm_index,
            )
            training["passive"][arm] = [contexts[i] + tail for i, tail in enumerate(tails.tolist())]
            print(f"seed={seed} arm={arm} collected", flush=True)
    return {
        "seed": seed,
        "training": training,
        "oracle": oracle,
        "public_prompts": prompts,
        "discovery_tokens": discovery.tolist(),
        "generated_training_tokens_per_strategy_per_arm": n * protocol["draws"],
        "shared_discovery_tokens": 2 * n * 24,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.out):
        if not args.score_only:
            if args.protocol is None:
                p.error("--protocol is required for collection")
            protocol = json.loads(args.protocol.read_text())
            identity = {
                "protocol": protocol,
                "source_hashes": {
                    name: file_digest(Path(__file__).parent / name)
                    for name in (
                        "prefix_experiment.py",
                        "uncertainty.py",
                        "scorer.py",
                        "tokens.py",
                        "evaluate.py",
                        "checkpoints.py",
                    )
                },
                "device": args.device,
            }
            manifest_path = args.out / "manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
                raise ValueError("source/protocol/device differs from existing experiment")
            atomic_json(manifest_path, identity)
            import torch
            import transformers
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                SynthIDTextWatermarkingConfig,
            )

            torch.set_num_threads(4)
            tokenizer = AutoTokenizer.from_pretrained(
                protocol["model"], revision=protocol["revision"], trust_remote_code=False
            )
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            model = (
                AutoModelForCausalLM.from_pretrained(
                    protocol["model"],
                    revision=protocol["revision"],
                    torch_dtype=torch.float32,
                    trust_remote_code=False,
                    use_safetensors=True,
                )
                .to(args.device)
                .eval()
            )
            config = SynthIDTextWatermarkingConfig(
                ngram_len=5, keys=evaluate.KEYS, skip_first_ngram_calls=True
            )
            runtime = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
                "model_revision": model.config._commit_hash,
                "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                "watermark": config.to_dict(),
            }
            runtime_path = args.out / "runtime.json"
            if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
                raise ValueError("runtime differs from existing experiment")
            atomic_json(runtime_path, runtime)
            for seed in protocol["seeds"]:
                path = args.out / f"seed-{seed}.json"
                if not path.exists():
                    data = collect_seed(args, protocol, seed, model, tokenizer, config)
                    atomic_json(path, data)
        identity = json.loads((args.out / "manifest.json").read_text())
        results = []
        for seed in identity["protocol"]["seeds"]:
            data = json.loads((args.out / f"seed-{seed}.json").read_text())
            results.extend(analyze_seed(data))
        atomic_json(
            args.out / "results.json",
            {
                "protocol": identity["protocol"],
                "results": results,
                "artifact_sha256": {
                    f"seed-{s}.json": file_digest(args.out / f"seed-{s}.json")
                    for s in identity["protocol"]["seeds"]
                },
                "analysis_source_sha256": file_digest(Path(__file__)),
            },
        )
        print(f"wrote {args.out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
