# -*- coding: utf-8 -*-
"""Table 5 summary for run_norm_ablation.sh phase 4 (oc_z clip x warmup grid).

Prints, per dataset: holdout RMSE / Score (per-engine mean) as mean ± SD over seeds for the
12 settings, the setting with the lowest mean validation RMSE (†), marginal means, and each
setting minus the clip/warmup-off oc_z arm, paired on the seeds both share.

    python experiments/summarize_clip_warmup.py experiments/norm_ablation_results/summary.jsonl
"""

import collections
import json
import statistics as st
import sys

from scipy import stats

RMSE, SCORE = "holdout_selby_valrmse_rmse", "holdout_selby_valrmse_score_engine_mean"
CLIPS, WARMUPS = (0.5, 1.0, 2.0, 5.0), (5, 10, 15)


def ms(xs):
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}"


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    m = st.mean(d)
    h = stats.t.ppf(0.975, len(d) - 1) * st.stdev(d) / len(d) ** 0.5
    return f"{m:+.2f} [{m - h:+.2f}, {m + h:+.2f}] (n={len(d)})"


def main(path):
    sys.stdout.reconfigure(encoding="utf-8")
    g = collections.defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("norm", "oc_z") == "oc_z":
            g[(r["dataset"], float(r["clip"]), int(r["warmup"]))][r["seed"]] = r
    for ds in ("FD002", "FD004"):
        grid = {(c, w): g[(ds, c, w)] for c in CLIPS for w in WARMUPS if g.get((ds, c, w))}
        if not grid:
            continue
        sel = min(grid, key=lambda k: st.mean(r["best_val_rmse"] for r in grid[k].values()))
        off = g.get((ds, 0.0, 0), {})
        print(f"\n## {ds}: holdout RMSE (mean ± SD; † = lowest mean validation RMSE)\n")
        print("| c \\ w | " + " | ".join(f"w = {w}" for w in WARMUPS) + " |")
        print("|---|" + "---|" * len(WARMUPS))
        for c in CLIPS:
            cells = []
            for w in WARMUPS:
                runs = grid.get((c, w))
                cells.append(ms([r[RMSE] for r in runs.values()]) + ("†" if (c, w) == sel else "") +
                             f" (n={len(runs)})" if runs else "-")
            print(f"| {c:g} | " + " | ".join(cells) + " |")
        means = [st.mean(r[RMSE] for r in v.values()) for v in grid.values()]
        sds = [st.stdev(r[RMSE] for r in v.values()) for v in grid.values() if len(v) > 1]
        print(f"\nrange of means {min(means):.2f}-{max(means):.2f}; seed SD {min(sds):.2f}-{max(sds):.2f} "
              f"(mean SD {st.mean(sds):.2f})")
        print("marginal clip:", {c: round(st.mean(st.mean(r[RMSE] for r in grid[(c, w)].values())
                                                for w in WARMUPS if (c, w) in grid), 2) for c in CLIPS})
        print("marginal warmup:", {w: round(st.mean(st.mean(r[RMSE] for r in grid[(c, w)].values())
                                                   for c in CLIPS if (c, w) in grid), 2) for w in WARMUPS})
        s = grid[sel]
        print(f"selected c={sel[0]:g}, w={sel[1]}: RMSE {ms([r[RMSE] for r in s.values()])}, "
              f"Score {ms([r[SCORE] for r in s.values()])}")
        if off:
            print(f"\nclip/warmup off (oc_z, clip 0, warmup 0): RMSE {ms([r[RMSE] for r in off.values()])} "
                  f"(n={len(off)})")
            print("\n| setting | ΔRMSE vs off, paired [95% CI] |\n|---|---|")
            for (c, w), runs in sorted(grid.items()):
                print(f"| c={c:g}, w={w}{' †' if (c, w) == sel else ''} | "
                      f"{paired({k: v[RMSE] for k, v in runs.items()}, {k: v[RMSE] for k, v in off.items()})} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/norm_ablation_results/summary.jsonl")
