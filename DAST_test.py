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
from DAST_visualization import get_test_window_oc_clusters, plot_training_results
import time
import math

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

def squeeze_last_dim(values):
    return values.squeeze(-1)

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
            pred = squeeze_last_dim(model(xb))
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP_ENABLED:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            train_loss_avg = total_loss / len(train_loader)

        
        # ── 評估 ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_np = squeeze_last_dim(model(testX.to(device))).cpu().numpy()

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
        final_pred_np = squeeze_last_dim(model(testX.to(device))).cpu().numpy()
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
