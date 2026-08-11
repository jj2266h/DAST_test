import random
import numpy as np
import scipy.io as sio
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import csv
import json
import argparse
from DAST_Network import DAST
import time
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.vq import kmeans as scipy_kmeans, vq as scipy_vq

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_array(path, key):
    return sio.loadmat(path)[key]

def append_experiment_log(csv_path, row):
    log_dir = os.path.dirname(csv_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    if file_exists:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = [
                {key: value for key, value in existing_row.items() if key is not None}
                for existing_row in reader
            ]

        missing_fields = [name for name in fieldnames if name not in existing_fieldnames]
        if missing_fields:
            fieldnames = existing_fieldnames + missing_fields
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
        else:
            fieldnames = existing_fieldnames

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_training_results(history, pred_np, testY_np, RUL_max, dataset, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"DAST Training Results — {dataset}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Train Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], color="steelblue")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["rmse"], color="tomato")
    best_epoch = int(np.argmin(history["rmse"])) + 1
    ax2.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax2.set_title("Test RMSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE (cycles)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Score
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history["score"], color="mediumseagreen")
    ax3.set_title("NASA Score")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score (lower is better)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Pred vs True scatter
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.scatter(testY_np * RUL_max, pred_np * RUL_max, s=8, alpha=0.5, color="steelblue", label="Predictions")
    lim = max(testY_np.max(), pred_np.max()) * RUL_max * 1.05
    ax4.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax4.set_title("Predicted vs True RUL (final epoch)")
    ax4.set_xlabel("True RUL (cycles)")
    ax4.set_ylabel("Predicted RUL (cycles)")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Pred vs True line
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(testY_np * RUL_max, label="True RUL", linewidth=1)
    ax5.plot(pred_np * RUL_max, label="Pred RUL", linewidth=1, alpha=0.8)
    ax5.set_title("RUL Comparison")
    ax5.set_xlabel("Sample")
    ax5.set_ylabel("RUL (cycles)")
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.5)

    save_path = os.path.join(save_dir, f"training_result_{dataset}_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"圖表已儲存至: {save_path}")
    return save_path

# ── 損失函數/指標 ──────────────────────────────────────────────────
def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

def compute_rmse(pred: np.ndarray, true: np.ndarray, rul_max: float = 1.0) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)) * rul_max)

def compute_mae(pred: np.ndarray, true: np.ndarray, rul_max: float = 1.0) -> float:
    return float(np.mean(np.abs(pred - true)) * rul_max)

def s_score(y_true, y_pred):
    diff = np.array(y_pred) - np.array(y_true)
    score = [(math.exp(-d/13.0) - 1.0) if d < 0 else (math.exp(d/10.0) - 1.0) for d in diff]
    return float(np.sum(score))

def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def fit_condition_clusters(train_raw, n_clusters):
    op_data = train_raw[:, 2:5].astype(float)
    scale = op_data.std(axis=0)
    scale[scale == 0] = 1.0
    centers, _ = scipy_kmeans(op_data / scale, n_clusters, iter=50, seed=42)
    centers = centers[np.argsort(centers[:, 0])]
    return centers, scale


def assign_condition_clusters(data, centers, scale):
    labels, _ = scipy_vq(data[:, 2:5].astype(float) / scale, centers)
    return labels


def get_test_window_oc_clusters(data_path, dataset, window_size, n_clusters):
    train_raw = np.loadtxt(f"{data_path}/train_{dataset}.txt")
    test_raw = np.loadtxt(f"{data_path}/test_{dataset}.txt")
    centers, scale = fit_condition_clusters(train_raw, n_clusters)

    unit_clusters = []
    for unit in range(1, int(np.max(test_raw[:, 0])) + 1):
        unit_rows = test_raw[test_raw[:, 0] == unit]
        window_rows = unit_rows[-window_size:] if len(unit_rows) >= window_size else unit_rows
        labels = assign_condition_clusters(window_rows, centers, scale)
        unit_clusters.append(int(np.bincount(labels, minlength=n_clusters).argmax()))
    return np.array(unit_clusters, dtype=int)


def plot_training_results(history, pred_np, testY_np, RUL_max, dataset, save_dir, oc_clusters=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)
    prefix = os.path.join(save_dir, f"{timestamp}_{dataset}")
    plot_paths = []

    true_rul = testY_np * RUL_max
    pred_rul = pred_np * RUL_max
    best_epoch = int(np.argmin(history["rmse"])) + 1

    def save_current(name):
        path = f"{prefix}_{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], color="steelblue", linewidth=1.5)
    ax.set_title(f"{dataset} Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_current("01_train_loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    if "val_rmse" in history:
        ax.plot(epochs, history["val_rmse"], color="steelblue", linewidth=1.3, label="Val RMSE")
    ax.plot(epochs, history["rmse"], color="tomato", linewidth=1.5, label="Test RMSE")
    ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax.set_title(f"{dataset} Val/Test RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (cycles)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    save_current("02_val_test_rmse")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["score"], color="mediumseagreen", linewidth=1.5)
    ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax.set_title(f"{dataset} NASA Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    save_current("03_nasa_score")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_rul, pred_rul, s=14, alpha=0.65, color="steelblue", label="Best checkpoint")
    lim = max(true_rul.max(), pred_rul.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"{dataset} Best Checkpoint Predicted vs True RUL")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    save_current("04_best_pred_vs_true_rul")

    order = np.argsort(true_rul)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true_rul[order], color="black", linewidth=1.6, label="True RUL")
    ax.plot(pred_rul[order], color="tomato", linewidth=1.3, alpha=0.9, label="Predicted RUL")
    ax.set_title(f"{dataset} Prediction Curve Sorted by True RUL")
    ax.set_xlabel("Samples sorted by true RUL")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    save_current("05_sorted_prediction_curve")

    if oc_clusters is not None:
        cluster_ids = np.array(sorted(np.unique(oc_clusters)))
        cluster_rmse = []
        cluster_score = []
        cluster_counts = []
        for cluster_id in cluster_ids:
            mask = oc_clusters == cluster_id
            cluster_rmse.append(compute_rmse(pred_np[mask], testY_np[mask], RUL_max))
            cluster_score.append(s_score(true_rul[mask], pred_rul[mask]))
            cluster_counts.append(int(mask.sum()))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = [f"OC {cluster_id}\n(n={count})" for cluster_id, count in zip(cluster_ids, cluster_counts)]
        axes[0].bar(labels, cluster_rmse, color="steelblue")
        axes[0].set_title("RMSE by OC Cluster")
        axes[0].set_ylabel("RMSE (cycles)")
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

        axes[1].bar(labels, cluster_score, color="mediumseagreen")
        axes[1].set_title("NASA Score by OC Cluster")
        axes[1].set_ylabel("Score (lower is better)")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.suptitle(f"{dataset} OC Cluster Metrics")
        save_current("06_oc_cluster_rmse_score")
    else:
        print("OC cluster labels unavailable; skipped OC cluster RMSE/Score plot.")

    print("Saved plots:")
    for path in plot_paths:
        print(f"  {path}")
    return plot_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Train DAST on a configured dataset.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--run-label", default="", help="Optional label for sweep runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    run_label = args.run_label

    # ── 讀取設定檔 ────────────────────────────────────────
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _ds  = config["cmapss"]
    _tr  = config["training"]
    _mdl = config["model"]

    seed = _tr.get("seed", 42)
    set_seed(seed)

    DATASET       = _ds["dataset"]
    RUL_max       = _ds["rul_max"]
    traindata_dir = _ds["output_path"]
    data_path     = _ds["data_path"]
    window_size   = _ds["window_size"]
    condition_cluster_count = _ds.get("condition_cluster_count", 6)
    BATCH_SIZE    = _tr["batch_size"]
    EPOCHS        = _tr["epochs"]
    LR            = _tr["learning_rate"]
    GRAD_CLIP_ENABLED = _tr.get("grad_clip_enabled", False)
    GRAD_CLIP_MAX_NORM = _tr.get("grad_clip_max_norm", 1.0)
    LR_WARMUP_ENABLED = _tr.get("lr_warmup_enabled", False)
    LR_WARMUP_EPOCHS = _tr.get("lr_warmup_epochs", 5)
    model_dir     = _tr["model_save_path"]
    run_id        = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_suffix    = f"{DATASET}_{run_label}_{run_id}" if run_label else f"{DATASET}_{run_id}"
    log_dir       = "紀錄"
    history_csv   = os.path.join(log_dir, f"training_history_{run_suffix}.csv")
    summary_csv   = os.path.join(log_dir, "experiment_log.csv")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    best_ckpt = os.path.join(model_dir, f"dast_{run_suffix}_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # ── 資料載入與處理 ─────────────────────────────────────
    trainX = load_array(f"{traindata_dir}/{DATASET}_window_size_trainX_new.mat", "train1X_new")
    trainY = load_array(f"{traindata_dir}/{DATASET}_window_size_trainY.mat", "train1Y").flatten()
    testX = load_array(f"{traindata_dir}/{DATASET}_window_size_testX_new.mat", "test1X_new")
    testY = load_array(f"{traindata_dir}/{DATASET}_window_size_testY.mat", "test1Y").flatten()

    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    trainY = torch.tensor(trainY, dtype=torch.float32).to(device)
    testY_np = testY.cpu().numpy() if isinstance(testY, torch.Tensor) else testY
    testY = torch.tensor(testY, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=BATCH_SIZE, shuffle=True)
    time_step  = trainX.shape[1]
    input_size = trainX.shape[2]
    print(f"Time step: {time_step}, Sensor count: {input_size}")
    # ── 模型建立 ─────────────────────────────────────────────
    
    model = DAST(
        dim_val_s=_mdl["dim_val"],
        dim_attn_s=_mdl["dim_attn"],
        dim_val_t=_mdl["dim_val"],
        dim_attn_t=_mdl["dim_attn"],
        dim_val=_mdl["dim_val"],
        dim_attn=_mdl["dim_attn"],
        time_step=time_step,
        input_size=input_size,
        dec_seq_len=_mdl["dec_seq_len"],
        out_seq_len=_mdl["out_seq_len"],
        n_encoder_layers=_mdl["n_encoder_layers"],
        n_decoder_layers=_mdl["n_decoder_layers"],
        n_heads=_mdl["n_heads"],
        dropout=_mdl["dropout"],
    ).to(device)

    optimizer = torch.optim.RAdam(model.parameters(), lr=LR)
    warmup_steps = max(1, LR_WARMUP_EPOCHS * len(train_loader))
    scheduler = None
    if LR_WARMUP_ENABLED:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )
    loss_fn = RMSELoss
    best_rmse=2000
    best_epoch = 0
    best_mae = None
    best_score = None
    loss_history = {"train_loss": [], "rmse": [], "mae": [], "score": []}
    start_time = time.time()
    print(f"Training history will be saved to: {history_csv}")
    print(
        f"Gradient clipping: {'on' if GRAD_CLIP_ENABLED else 'off'} "
        f"(max_norm={GRAD_CLIP_MAX_NORM})"
    )
    print(
        f"LR warmup: {'on' if LR_WARMUP_ENABLED else 'off'} "
        f"(warmup_epochs={LR_WARMUP_EPOCHS})"
    )

    # ── 訓練迴圈 ────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP_ENABLED:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        
        # ── 評估 ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_np = model(testX.to(device)).squeeze().cpu().numpy()
        train_loss_avg = total_loss / len(train_loader)
        rmse        = compute_rmse(pred_np, testY_np, RUL_max)
        mae         = compute_mae(pred_np, testY_np, RUL_max)
        score_value = s_score(testY_np * RUL_max, pred_np * RUL_max)
        loss_history["train_loss"].append(train_loss_avg)
        loss_history["rmse"].append(rmse)
        loss_history["mae"].append(mae)
        loss_history["score"].append(score_value)
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        is_best = rmse < best_rmse
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss_avg:.4f} "
              f"| Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | Score: {score_value:.1f}")
        append_experiment_log(history_csv, {
            "run_id": run_id,
            "run_label": run_label,
            "config_path": config_path,
            "dataset": DATASET,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "rmse": rmse,
            "mae": mae,
            "score": score_value,
            "epoch_time_sec": epoch_time,
            "elapsed_time_sec": elapsed_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": BATCH_SIZE,
            "grad_clip_enabled": int(GRAD_CLIP_ENABLED),
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "lr_warmup_enabled": int(LR_WARMUP_ENABLED),
            "lr_warmup_epochs": LR_WARMUP_EPOCHS,
            "is_best": int(is_best),
        })
        if is_best:
            best_rmse = rmse
            best_epoch = epoch
            best_mae = mae
            best_score = score_value
            torch.save(model.state_dict(), best_ckpt)

    # ── 儲存模型 ────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(model_dir, f"dast_{run_suffix}.pth"))
    model.to(device)
    print("模型已儲存")

    print(f"\nBest RMSE (HI): {best_rmse:.5f}")
    print("模型已儲存")

    # ── 視覺化 ──────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        final_pred_np = model(testX.to(device)).squeeze().cpu().numpy()
    oc_clusters = get_test_window_oc_clusters(
        data_path, DATASET, window_size, condition_cluster_count
    )
    plot_path = plot_training_results(
        loss_history,
        final_pred_np,
        testY_np,
        RUL_max,
        DATASET,
        save_dir="plots",
        oc_clusters=oc_clusters,
    )

    append_experiment_log(summary_csv, {
        "run_id": run_id,
        "run_label": run_label,
        "config_path": config_path,
        "dataset": DATASET,
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "grad_clip_enabled": int(GRAD_CLIP_ENABLED),
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "lr_warmup_enabled": int(LR_WARMUP_ENABLED),
        "lr_warmup_epochs": LR_WARMUP_EPOCHS,
        "rul_max": RUL_max,
        "time_step": time_step,
        "input_size": input_size,
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "best_mae": best_mae,
        "best_score": best_score,
        "final_train_loss": loss_history["train_loss"][-1],
        "final_rmse": loss_history["rmse"][-1],
        "final_mae": loss_history["mae"][-1],
        "final_score": loss_history["score"][-1],
        "elapsed_time_sec": time.time() - start_time,
        "history_csv": history_csv,
        "plot_path": ";".join(plot_path),
        "best_checkpoint": best_ckpt,
    })
    print(f"Experiment summary saved to: {summary_csv}")

if __name__ == '__main__':
    main()

# python data_process.py
