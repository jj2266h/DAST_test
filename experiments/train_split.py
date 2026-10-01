# -*- coding: utf-8 -*-
"""Train DAST with engine-disjoint validation. The official test set is never read.

Differs from `DAST_test.py` in exactly one way that matters: `is_best` is decided on
the validation split, not on the test set. Both val-RMSE and val-Score selection are
tracked in the same run so the two rules can be compared without retraining.
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DAST_Network import DAST
from experiments.prepare_splits import NORMS, split_stem


def parse_args():
    p = argparse.ArgumentParser(description="Train DAST on engine-disjoint splits.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--dataset", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--norm", default="oc_z", choices=NORMS)
    p.add_argument("--split-dir", default="split_dataset")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--clip", type=float, default=None, help="Max grad norm; 0 disables clipping.")
    p.add_argument("--warmup", type=int, default=None, help="Warmup epochs; 0 disables warmup.")
    p.add_argument("--out-dir", default="experiments/idea1_results")
    p.add_argument("--eval-holdout", action="store_true",
                   help="Evaluate the selected checkpoints on the holdout split at the end.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(split_dir, dataset, norm, variant, name):
    stem = f"{split_dir}/{split_stem(dataset, norm, variant)}_{name}"
    X = sio.loadmat(f"{stem}X.mat")["X"]
    Y = sio.loadmat(f"{stem}Y.mat")["Y"].flatten()
    E = sio.loadmat(f"{stem}E.mat")["E"].flatten()
    return X, Y, E


def rmse_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def nasa_score(true_cycles, pred_cycles):
    d = np.asarray(pred_cycles) - np.asarray(true_cycles)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0), np.exp(d / 10.0)) - 1.0))


def evaluate(model, X, Y, E, rul_max, device, batch=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            preds.append(model(xb).squeeze(-1).cpu().numpy().ravel())
    pred = np.concatenate(preds) * rul_max
    true = Y * rul_max
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    # engine-level aggregation: mean NASA loss within engine, then mean across engines
    per_engine = [np.mean(np.where((pred[E == u] - true[E == u]) < 0,
                                   np.exp(-(pred[E == u] - true[E == u]) / 13.0),
                                   np.exp((pred[E == u] - true[E == u]) / 10.0)) - 1.0)
                  for u in np.unique(E)]
    per_engine = np.array(per_engine)
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(pred - true))),
        "score_sum": nasa_score(true, pred),
        "score_engine_mean": float(per_engine.mean()),
        "score_engine_p90": float(np.quantile(per_engine, 0.90)),
        "cap_rate": float(np.mean(pred >= rul_max * 0.98)),
    }


def main():
    args = parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    tr_cfg, mdl_cfg = cfg["training"], cfg["model"]
    rul_max = float(cfg["cmapss"]["rul_max"])

    epochs = args.epochs or tr_cfg["epochs"]
    # 0 means "off"; without a CLI value the config's enabled flags decide.
    if args.clip is not None:
        clip = args.clip
    else:
        clip = tr_cfg["grad_clip_max_norm"] if tr_cfg.get("grad_clip_enabled", True) else 0.0
    if args.warmup is not None:
        warmup = args.warmup
    else:
        warmup = tr_cfg["lr_warmup_epochs"] if tr_cfg.get("lr_warmup_enabled", False) else 0
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trX, trY, _ = load_split(args.split_dir, args.dataset, args.norm, args.variant, "train")
    vaX, vaY, vaE = load_split(args.split_dir, args.dataset, args.norm, args.variant, "val")
    print(f"{args.dataset}/{args.norm}/{args.variant} seed={args.seed} clip={clip} warmup={warmup} device={device}")
    print(f"  train {trX.shape} | val {vaX.shape} ({len(np.unique(vaE))} engines)")

    loader = DataLoader(
        TensorDataset(torch.tensor(trX, dtype=torch.float32),
                      torch.tensor(trY, dtype=torch.float32)),
        batch_size=tr_cfg["batch_size"], shuffle=True)

    model = DAST(
        dim_val_s=mdl_cfg["dim_val"], dim_attn_s=mdl_cfg["dim_attn"],
        dim_val_t=mdl_cfg["dim_val"], dim_attn_t=mdl_cfg["dim_attn"],
        dim_val=mdl_cfg["dim_val"], dim_attn=mdl_cfg["dim_attn"],
        time_step=trX.shape[1], input_size=trX.shape[2],
        dec_seq_len=mdl_cfg["dec_seq_len"], out_seq_len=mdl_cfg["out_seq_len"],
        n_encoder_layers=mdl_cfg["n_encoder_layers"],
        n_decoder_layers=mdl_cfg["n_decoder_layers"],
        n_heads=mdl_cfg["n_heads"], dropout=mdl_cfg["dropout"],
    ).to(device)

    opt = torch.optim.RAdam(model.parameters(), lr=tr_cfg["learning_rate"])
    steps = max(1, warmup * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / steps)) if warmup > 0 else None

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.norm}_{args.variant}_clip{clip}_warmup{warmup}_seed{args.seed}"
    hist_path = os.path.join(args.out_dir, f"history_{tag}.csv")
    ck_rmse = os.path.join(args.out_dir, f"best_valrmse_{tag}.pth")
    ck_score = os.path.join(args.out_dir, f"best_valscore_{tag}.pth")

    best = {"rmse": float("inf"), "score": float("inf")}
    rows = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = rmse_loss(model(xb).squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched is not None:
                sched.step()
            tot += loss.item()

        m = evaluate(model, vaX, vaY, vaE, rul_max, device)
        is_best_rmse = m["rmse"] < best["rmse"]
        is_best_score = m["score_engine_mean"] < best["score"]
        if is_best_rmse:
            best["rmse"] = m["rmse"]
            torch.save(model.state_dict(), ck_rmse)
        if is_best_score:
            best["score"] = m["score_engine_mean"]
            torch.save(model.state_dict(), ck_score)

        rows.append({
            "epoch": ep, "train_loss": tot / len(loader),
            "val_rmse": m["rmse"], "val_mae": m["mae"],
            "val_score_sum": m["score_sum"],
            "val_score_engine_mean": m["score_engine_mean"],
            "val_score_engine_p90": m["score_engine_p90"],
            "val_cap_rate": m["cap_rate"],
            "is_best_rmse": int(is_best_rmse), "is_best_score": int(is_best_score),
            "elapsed_sec": time.time() - t0,
        })
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d} | train {tot/len(loader):.4f} | val RMSE {m['rmse']:.3f} "
                  f"| val Score/engine {m['score_engine_mean']:.2f} | cap {m['cap_rate']:.1%}")

    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"tag": tag, "dataset": args.dataset, "norm": args.norm, "variant": args.variant,
               "seed": args.seed, "clip": clip, "warmup": warmup, "epochs": epochs,
               "best_val_rmse": best["rmse"], "best_val_score_engine_mean": best["score"],
               "train_minutes": (time.time() - t0) / 60.0}

    if args.eval_holdout:
        hoX, hoY, hoE = load_split(args.split_dir, args.dataset, args.norm, args.variant, "holdout")
        for rule, ck in (("selby_valrmse", ck_rmse), ("selby_valscore", ck_score)):
            model.load_state_dict(torch.load(ck, map_location=device))
            hm = evaluate(model, hoX, hoY, hoE, rul_max, device)
            summary.update({f"holdout_{rule}_{k}": v for k, v in hm.items()})
            print(f"  HOLDOUT [{rule}] RMSE {hm['rmse']:.3f} | "
                  f"Score/engine {hm['score_engine_mean']:.2f} | p90 {hm['score_engine_p90']:.2f}")

    sum_path = os.path.join(args.out_dir, "summary.jsonl")
    with open(sum_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"  -> {hist_path}\n  -> {sum_path}")


if __name__ == "__main__":
    main()
