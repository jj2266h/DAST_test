# -*- coding: utf-8 -*-
"""Summarize run_norm_ablation.sh results for Table 3 (RQ2) and the RQ3 on/off arms.

Holdout metrics use the checkpoint selected on validation RMSE. Differences are paired
by seed against the reference arm; the 95% CI is t-based. With 5 seeds report these as
descriptive effect sizes, not significance tests.

    python experiments/summarize_norm_ablation.py experiments/norm_ablation_results/summary.jsonl

With --cut, report the one-random-cut-per-engine metrics written by reeval_holdout.py:

    python experiments/summarize_norm_ablation.py experiments/norm_ablation_results/summary_cut.jsonl --cut
"""

import collections
import json
import statistics as st
import sys

from scipy import stats

METRICS = (("RMSE", "holdout_selby_valrmse_rmse"),
           ("Score", "holdout_selby_valrmse_score_sum"))
CUT_METRICS = (("RMSE", "holdout_cut_rmse"),
               ("Score", "holdout_cut_score_sum"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def arm(r):
    return (r["dataset"], r.get("norm", "oc_z"), float(r["clip"]), int(r["warmup"]))


def mean_sd(xs):
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def paired(a, b, key):
    """Mean of (a - b) over shared seeds, with a t-based 95% CI."""
    seeds = sorted(set(a) & set(b))
    d = [a[s][key] - b[s][key] for s in seeds]
    if len(d) < 2:
        return f"n={len(d)}"
    m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return f"{m:+.2f} [{m - h:+.2f}, {m + h:+.2f}] (n={len(d)})"


def main(path, metrics=METRICS):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp950
    groups = collections.defaultdict(dict)
    for r in load(path):
        groups[arm(r)][r["seed"]] = r

    print("## Arms (holdout, mean ± SD over seeds)\n")
    print("| Dataset | Norm | clip | warmup | n | RMSE | Score |")
    print("|---|---|---|---|---|---|---|")
    for (ds, norm, c, w), runs in sorted(groups.items()):
        cols = [mean_sd([r[k] for r in runs.values()]) for _, k in metrics]
        print(f"| {ds} | {norm} | {c:g} | {w} | {len(runs)} | " + " | ".join(cols) + " |")

    print("\n## Paired differences vs reference (negative = lower error)\n")
    print("| Dataset | Comparison | ΔRMSE [95% CI] | ΔScore [95% CI] |")
    print("|---|---|---|---|")
    datasets = sorted({k[0] for k in groups})
    for ds in datasets:
        base = {n: groups.get((ds, n, 0.0, 0))
                for n in ("oc_z", "oc_minmax", "global_z", "global_minmax")}
        pairs = [("oc_z − global_z", base["oc_z"], base["global_z"]),
                 ("oc_z − global_minmax", base["oc_z"], base["global_minmax"]),
                 ("oc_minmax − global_minmax", base["oc_minmax"], base["global_minmax"]),
                 ("oc_z − oc_minmax", base["oc_z"], base["oc_minmax"]),
                 ("global_z − global_minmax", base["global_z"], base["global_minmax"])]
        for (c, w), name in (((0.0, 5), "warmup"), ((5.0, 0), "clip"), ((5.0, 5), "warmup+clip")):
            pairs.append((f"oc_z +{name} − oc_z", groups.get((ds, "oc_z", c, w)), base["oc_z"]))
        for label, a, b in pairs:
            if a and b:
                print(f"| {ds} | {label} | " + " | ".join(paired(a, b, k) for _, k in metrics) + " |")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--cut"]
    main(args[0] if args else "experiments/norm_ablation_results/summary.jsonl",
         CUT_METRICS if "--cut" in sys.argv else METRICS)
