"""Conservative contextual contrasts with approximate uncertainty.

No keys, logits, or evaluation labels are accepted by this estimator. A normal
approximation to smoothed log probability ratios gives a variance-dependent
shrinkage score. This is a ranking heuristic, not calibrated posterior inference.
"""

from __future__ import annotations

import math

from scorer import context_key


def contrast(w_count, w_total, b_count, b_total, support, alpha=0.5, prior_variance=1.0):
    """Estimate a signed log ratio, delta-method SE, and shrunken ranking score."""
    if alpha <= 0 or prior_variance <= 0 or support < 1:
        raise ValueError("alpha, prior_variance and support must be positive")
    if not (0 <= w_count <= w_total and 0 <= b_count <= b_total):
        raise ValueError("counts must lie between zero and their totals")
    if w_total == 0 or b_total == 0:
        return None
    wa, ba = w_count + alpha, b_count + alpha
    wn, bn = w_total + alpha * support, b_total + alpha * support
    log_ratio = math.log(wa / wn) - math.log(ba / bn)
    variance = max(0.0, 1 / wa - 1 / wn + 1 / ba - 1 / bn)
    reliability = prior_variance / (prior_variance + variance)
    return {
        "score": log_ratio * reliability,
        "log_ratio": log_ratio,
        "standard_error": math.sqrt(variance),
        "reliability": reliability,
        "watermarked_count": w_count,
        "baseline_count": b_count,
    }


def build_scorer(wm, base, context_len, alpha=0.5, prior_variance=1.0):
    """Retain both signed tails over shared contexts; unseen contexts abstain."""
    if context_len < 1:
        raise ValueError("context_len must be positive")
    if alpha <= 0 or prior_variance <= 0:
        raise ValueError("alpha and prior_variance must be positive")
    wmap, wt, wu = wm
    bmap, bt, bu = base
    if not wu or not bu:
        raise ValueError("both corpora must be nonempty")
    table = {}
    for ctx in sorted(wmap.keys() & bmap.keys()):
        support = sorted(wmap[ctx].keys() | bmap[ctx].keys())
        table[context_key(ctx)] = [
            {
                "token": token,
                **contrast(
                    wmap[ctx].get(token, 0),
                    wt[ctx],
                    bmap[ctx].get(token, 0),
                    bt[ctx],
                    len(support),
                    alpha,
                    prior_variance,
                ),
            }
            for token in support
        ]
    return {
        "config": {
            "context_len": context_len,
            "alpha": alpha,
            "prior_variance": prior_variance,
            "baseline_fallback": "abstain",
            "uncertainty": "delta-method approximation; ranking heuristic only",
        },
        "scorer": table,
    }
