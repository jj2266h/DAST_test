# -*- coding: utf-8 -*-
"""
DAST 訓練腳本 — PHM2012 (PRONOSTIA/FEMTO) HI 預測版本

架構參考 DAST_test.py；模型從 DAST_Network 匯入，架構零修改。
與 CMAPSS 版本的差異：資料載入、config 區段、無 RUL_max、無 S-score。
"""
import os
import numpy as np
import scipy.io as sio
import torch
import csv
import time
import json
from torch.utils.data import DataLoader, TensorDataset
from DAST_Network import DAST

# ── 工況與軸承對照表 ──────────────────────────────────────────
# Condition 1: 1800 rpm / 4000 N
# Condition 2: 1650 rpm / 4200 N
# Condition 3: 1500 rpm / 5000 N
CONDITION_MAP = {
    "1": {
        "train": ["Bearing1_1", "Bearing1_2"],
        "test":  ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"],
    },
    "2": {
        "train": ["Bearing2_1", "Bearing2_2"],
        "test":  ["Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"],
    },
    "3": {
        "train": ["Bearing3_1", "Bearing3_2"],
        "test":  ["Bearing3_3"],
    },
}

# ── 工具函數 ──────────────────────────────────────────────────

def append_experiment_log(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_femto_split(data_root: str, condition: str, is_train: bool):
    """
    依工況 condition 和 is_train 旗標，
    載入對應軸承的 .mat 檔並合併回傳 X (N, W, 16) 與 Y (N,)。
    """
    bearings = CONDITION_MAP[condition]["train" if is_train else "test"]
    x_key    = "PHM_trainX" if is_train else "PHM_testX"
    y_key    = "PHM_trainY" if is_train else "PHM_testY"

    x_list, y_list = [], []
    for bearing in bearings:
        x_path = os.path.join(data_root, f"{bearing}_X.mat")
        y_path = os.path.join(data_root, f"{bearing}_Y.mat")
        if not os.path.exists(x_path):
            print(f"  Warning: {x_path} not found, skipping.")
            continue
        x_arr = sio.loadmat(x_path)[x_key].astype(np.float32)
        y_arr = sio.loadmat(y_path)[y_key].flatten().astype(np.float32)
        x_list.append(x_arr)
        y_list.append(y_arr)
        label = "train" if is_train else "test "
        print(f"  [{label}] {bearing}  {x_arr.shape}")

    if not x_list:
        raise FileNotFoundError(
            f"No {'train' if is_train else 'test'} files for condition {condition} in {data_root}"
        )
    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)


def plot_training_results(history, pred_np, testY_np, save_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        print(f"Matplotlib 無法載入，略過訓練圖輸出: {exc}")
        return

    epochs    = range(1, len(history["train_loss"]) + 1)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("DAST Training Results — PHM2012 (FEMTO) HI Prediction",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], marker="o", color="steelblue")
    ax1.set_title("Train Loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["rmse"], marker="s", color="tomato")
    best_epoch = int(np.argmin(history["rmse"])) + 1
    ax2.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7,
                label=f"Best epoch {best_epoch}")
    ax2.set_title("Test RMSE (HI)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history["mae"], marker="^", color="mediumseagreen")
    ax3.set_title("Test MAE (HI)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MAE")
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.scatter(testY_np, pred_np, s=8, alpha=0.5, color="steelblue",
                label="Predictions")
    ax4.plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfect prediction")
    ax4.set_title("Predicted vs True HI (final epoch)")
    ax4.set_xlabel("True HI")
    ax4.set_ylabel("Predicted HI")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    ax5 = fig.add_subplot(gs[1, 2])
    sort_idx = np.argsort(testY_np)
    ax5.plot(testY_np[sort_idx], label="True HI",  linewidth=1)
    ax5.plot(pred_np[sort_idx],  label="Pred HI",  linewidth=1, alpha=0.8)
    ax5.set_title("Sorted HI Comparison")
    ax5.set_xlabel("Sample (sorted)")
    ax5.set_ylabel("HI")
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.5)

    save_path = os.path.join(save_dir, f"training_result_FEMTO_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"圖表已儲存至: {save_path}")


# ── 主流程 ────────────────────────────────────────────────────

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    _ds  = config["femto"]
    _tr  = config["training"]
    _mdl = config["model"]

    data_root  = _ds["output_path"]
    CONDITION  = _ds["condition"]          # "1" / "2" / "3"
    BATCH_SIZE = _tr["batch_size"]
    EPOCHS     = _tr["epochs"]
    LR         = _tr["learning_rate"]
    model_dir  = _tr["model_save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}  工況: Condition {CONDITION}\n")

    # ── 資料載入 ──
    print("載入訓練資料...")
    trainX_np, trainY_np = load_femto_split(data_root, CONDITION, is_train=True)
    print(f"\n載入測試資料...")
    testX_np,  testY_np  = load_femto_split(data_root, CONDITION, is_train=False)
    print(f"\nTrain: {trainX_np.shape}  Test: {testX_np.shape}\n")

    trainX = torch.tensor(trainX_np, dtype=torch.float32).to(device)
    testX  = torch.tensor(testX_np,  dtype=torch.float32).to(device)
    trainY = torch.tensor(trainY_np, dtype=torch.float32).to(device)
    testY  = torch.tensor(testY_np,  dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(trainX, trainY), batch_size=BATCH_SIZE, shuffle=True
    )

    time_step  = trainX.shape[1]   # W  = 40
    input_size = trainX.shape[2]   # F  = 16
    print(f"Time step: {time_step}, Input size: {input_size}")

    # ── 模型（架構與 CMAPSS 完全一致）──
    model = DAST(
        dim_val_s=_mdl["dim_val"],   dim_attn_s=_mdl["dim_attn"],
        dim_val_t=_mdl["dim_val"],   dim_attn_t=_mdl["dim_attn"],
        dim_val=_mdl["dim_val"],     dim_attn=_mdl["dim_attn"],
        time_step=time_step,
        input_size=input_size,
        dec_seq_len=_mdl["dec_seq_len"],
        out_seq_len=_mdl["out_seq_len"],
        n_encoder_layers=_mdl["n_encoder_layers"],
        n_decoder_layers=_mdl["n_decoder_layers"],
        n_heads=_mdl["n_heads"],
        dropout=_mdl["dropout"],
    ).to(device)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.RAdam(model.parameters(), lr=LR)
    loss_fn   = torch.nn.MSELoss()
    best_rmse = 1e9
    loss_history = {"train_loss": [], "rmse": [], "mae": []}

    # ── 訓練迴圈 ──
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── 評估 ──
        model.eval()
        with torch.no_grad():
            pred_np = model(testX).squeeze().cpu().numpy()

        rmse = float(np.sqrt(np.mean((pred_np - testY_np) ** 2)))
        mae  = float(np.mean(np.abs(pred_np - testY_np)))

        loss_history["train_loss"].append(total_loss / len(train_loader))
        loss_history["rmse"].append(rmse)
        loss_history["mae"].append(mae)

        print(f"Epoch {epoch:3d} | Train Loss: {total_loss/len(train_loader):.5f} "
              f"| Test RMSE: {rmse:.5f} | Test MAE: {mae:.5f}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f"dast_FEMTO_cond{CONDITION}_best.pth")

    # ── 儲存與視覺化 ──
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, f'dast_FEMTO_cond{CONDITION}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth'),
    )
    print(f"\nBest RMSE (HI): {best_rmse:.5f}")
    print("模型已儲存")

    model.eval()
    with torch.no_grad():
        final_pred = model(testX).squeeze().cpu().numpy()
    plot_training_results(loss_history, final_pred, testY_np, save_dir="plots")


if __name__ == "__main__":
    main()
