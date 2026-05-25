import random
import numpy as np
import scipy.io as sio
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import csv
import json
from DAST_Network import DAST
import time
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
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

def main():
    # ── 讀取設定檔 ────────────────────────────────────────
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    _ds  = config["cmapss"]
    _tr  = config["training"]
    _mdl = config["model"]

    set_seed(_tr.get("seed", 42))

    DATASET       = _ds["dataset"]
    RUL_max       = _ds["rul_max"]
    traindata_dir = _ds["output_path"]
    BATCH_SIZE    = _tr["batch_size"]
    EPOCHS        = _tr["epochs"]
    LR            = _tr["learning_rate"]
    GRAD_CLIP_ENABLED = _tr.get("grad_clip_enabled", False)
    GRAD_CLIP_MAX_NORM = _tr.get("grad_clip_max_norm", 1.0)
    LR_WARMUP_ENABLED = _tr.get("lr_warmup_enabled", False)
    LR_WARMUP_EPOCHS = _tr.get("lr_warmup_epochs", 5)
    model_dir     = _tr["model_save_path"]
    run_id        = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir       = "紀錄"
    history_csv   = os.path.join(log_dir, f"training_history_{DATASET}_{run_id}.csv")
    summary_csv   = os.path.join(log_dir, "experiment_log.csv")
    os.makedirs(log_dir, exist_ok=True)

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
            "dataset": DATASET,
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
            torch.save(model.state_dict(), f'dast_{DATASET}_best.pth')

    # ── 儲存模型 ────────────────────────────────────────────
    torch.save(model.state_dict(), f'{model_dir}/dast_{DATASET}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth')
    model.to(device)
    print("模型已儲存")

    print(f"\nBest RMSE (HI): {best_rmse:.5f}")
    print("模型已儲存")

    # ── 視覺化 ──────────────────────────────────────────────
    best_ckpt = f"dast_{DATASET}_best.pth"
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        final_pred_np = model(testX.to(device)).squeeze().cpu().numpy()
    plot_path = plot_training_results(loss_history, final_pred_np, testY_np, RUL_max, DATASET, save_dir="plots")

    append_experiment_log(summary_csv, {
        "run_id": run_id,
        "dataset": DATASET,
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
        "plot_path": plot_path,
    })
    print(f"Experiment summary saved to: {summary_csv}")

if __name__ == '__main__':
    main()

# python data_process.py
