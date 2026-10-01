# Builds column-width figures and Table 4 data for the CIIE draft. Run from repo root.
import csv, json, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7})

# ---- Table 4: clean-protocol clip x warmup grid (OC z-score, val-selected, holdout) ----
SUMMARY = "../../../server_results/experiments/clean_sweep_results/summary.jsonl"
rows = [json.loads(l) for l in open(SUMMARY, encoding="utf-8")]
out = []
for d in ("FD002", "FD004"):
    for c in (0.5, 1.0, 2.0, 5.0):
        for w in (5, 10, 15):
            g = [r for r in rows if r["dataset"] == d and r["clip"] == c and r["warmup"] == w]
            f = lambda k: [r[k] for r in g]
            out.append({"dataset": d, "clip": c, "warmup": w, "n": len(g),
                        "val_rmse_mean": st.mean(f("best_val_rmse")),
                        "holdout_rmse_mean": st.mean(f("holdout_selby_valrmse_rmse")),
                        "holdout_rmse_sd": st.stdev(f("holdout_selby_valrmse_rmse")),
                        "holdout_score_mean": st.mean(f("holdout_selby_valrmse_score_engine_mean")),
                        "holdout_score_sd": st.stdev(f("holdout_selby_valrmse_score_engine_mean"))})
with open("paper/data/table4_clip_warmup_holdout.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
for d in ("FD002", "FD004"):
    sub = [r for r in out if r["dataset"] == d]
    best = min(sub, key=lambda r: r["val_rmse_mean"])
    print(d, "val-selected config", best["clip"], best["warmup"],
          f"holdout RMSE {best['holdout_rmse_mean']:.2f}+-{best['holdout_rmse_sd']:.2f}",
          f"Score {best['holdout_score_mean']:.2f}+-{best['holdout_score_sd']:.2f}")
    for key, levels in (("clip", (0.5, 1.0, 2.0, 5.0)), ("warmup", (5, 10, 15))):
        print("  marginal", key, {l: round(st.mean([r["holdout_rmse_mean"] for r in sub if r[key] == l]), 2) for l in levels})
    hr = [r["holdout_rmse_mean"] for r in sub]
    print(f"  range {min(hr):.2f}-{max(hr):.2f}, SD range {min(r['holdout_rmse_sd'] for r in sub):.2f}-{max(r['holdout_rmse_sd'] for r in sub):.2f}")

# ---- Fig 3 (column width): RQ1 metrics ----
R = {}
for r in csv.DictReader(open("paper/data/rq1_oc_variance.csv")):
    R.setdefault(r["dataset"], []).append(r)
sensors = [r["sensor"] for r in R["FD002"]]
x = np.arange(len(sensors))
fig, axes = plt.subplots(4, 1, figsize=(3.4, 5.6), sharex=True)
panels = [("FD002", "eta2_window", "(a) FD002: window-level OC-explained variance"),
          ("FD004", "eta2_window", "(b) FD004: window-level OC-explained variance"),
          ("FD002", "abs_rho_rul", "(c) FD002: |Spearman rho| with RUL"),
          ("FD004", "abs_rho_rul", "(d) FD004: |Spearman rho| with RUL")]
for ax, (d, m, title) in zip(axes, panels):
    raw = [float(r[f"{m}_raw_or_global"]) for r in R[d]]
    oc = [float(r[f"{m}_ocz"]) for r in R[d]]
    ax.bar(x - 0.2, raw, 0.4, color="#9aa5b1", label="Raw / global normalization")
    ax.bar(x + 0.2, oc, 0.4, color="#2f6db3", label="OC-aware z-score")
    ax.set_title(title, fontsize=7, loc="left")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=6)
axes[-1].legend(fontsize=6, loc="upper right", framealpha=0.9)
axes[-1].set_xticks(x); axes[-1].set_xticklabels(sensors, fontsize=6)
fig.tight_layout(h_pad=0.6)
fig.savefig("paper/figures/fig3_rq1_column.png", dpi=300)

# ---- Fig 1 (column width): framework ----
fig, ax = plt.subplots(figsize=(3.4, 3.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 11.6); ax.axis("off")
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=fc, ec="#333", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.6)
def arrow(x1, y1, x2, y2, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=0.7, ls="--" if dashed else "-", color="#333"))
box(1.2, 10.5, 7.6, 0.8, "C-MAPSS training file: engine-disjoint split\n(train 70% / validation 15% / holdout 15%)", "#eef2f7")
ax.text(2.3, 9.95, "Training engines", fontsize=6.5, ha="center", weight="bold")
ax.text(7.7, 9.95, "Validation / holdout engines", fontsize=6.5, ha="center", weight="bold")
arrow(3.5, 10.5, 2.3, 10.15); arrow(6.5, 10.5, 7.7, 10.15)
L, Rx, W, H = 0.0, 5.6, 4.4, 0.85
box(L, 8.75, W, H, "Fit K-means on (op1, op2, op3)\n-> OC centers", "#dbe7f5")
box(Rx, 8.75, W, H, "Assign OC with\ntrain-fitted centers", "#f3f3f3")
box(L, 7.45, W, H, "Estimate $\\mu_{c,j}$, $\\sigma_{c,j}$\nper OC $c$ and sensor $j$", "#dbe7f5")
box(Rx, 7.45, W, H, "Apply train $\\mu$, $\\sigma$\n(no re-estimation)", "#f3f3f3")
arrow(2.2, 8.75, 2.2, 8.3); arrow(7.8, 8.75, 7.8, 8.3)
arrow(4.4, 9.17, 5.6, 9.17, dashed=True); arrow(4.4, 7.87, 5.6, 7.87, dashed=True)
box(0.8, 5.95, 8.4, 0.95, "OC-aware z-score -> 14 sensors -> sliding window (T = 40)\n+ window slope and mean rows (T + 2 = 42)", "#e7f0e3")
arrow(2.2, 7.45, 3.5, 6.9); arrow(7.8, 7.45, 6.5, 6.9)
box(0.3, 4.35, 4.4, 0.9, "Sensor encoder\n(attention over sensors)", "#fbeede")
box(5.3, 4.35, 4.4, 0.9, "Time-step encoder\n(attention over time steps)", "#fbeede")
arrow(3.8, 5.95, 2.65, 5.25); arrow(6.2, 5.95, 7.35, 5.25)
box(2.5, 2.95, 5.0, 0.8, "Feature fusion", "#fbeede")
arrow(2.65, 4.35, 4.0, 3.75); arrow(7.35, 4.35, 6.0, 3.75)
box(2.5, 1.6, 5.0, 0.8, "Decoder -> RUL prediction", "#fbeede")
arrow(5, 2.95, 5, 2.4)
box(1.0, 0.1, 8.0, 0.9, "Model selection on validation RMSE;\nevaluation on holdout engines (RMSE, NASA Score)", "#eef2f7")
arrow(5, 1.6, 5, 1.0)
ax.add_patch(FancyBboxPatch((0.1, 1.45), 9.8, 4.0, boxstyle="round,pad=0.05", fc="none", ec="#b07a30", lw=0.6, ls=":"))
ax.text(0.25, 2.0, "DAST\n(unchanged)", fontsize=5.6, ha="left", style="italic", color="#7a4a10")
fig.savefig("paper/figures/fig1_framework.png", dpi=300, bbox_inches="tight")
print("figures saved")
