# -*- coding: utf-8 -*-
"""
DAST 訓練腳本 — PHM2012 (PRONOSTIA/FEMTO) HI 預測版本

架構參考 DAST_test.py；模型從 DAST_Network 匯入，架構零修改。
與 CMAPSS 版本的差異：資料載入、config 區段、無 RUL_max、無 S-score。
"""
import os
import random
import numpy as np
import scipy.io as sio
import torch
import csv
import time
import json
from torch.utils.data import DataLoader, TensorDataset
from DAST_Network import DAST


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    """載入並合併指定工況的所有軸承資料，回傳 (X, Y)。"""
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


def load_femto_test_bearings(data_root: str, condition: str):
    """回傳 list of (bearing_name, X_np, Y_np)，每個 test 軸承各自獨立。"""
    bearings = CONDITION_MAP[condition]["test"]
    result = []
    for bearing in bearings:
        x_path = os.path.join(data_root, f"{bearing}_X.mat")
        y_path = os.path.join(data_root, f"{bearing}_Y.mat")
        if not os.path.exists(x_path):
            print(f"  Warning: {x_path} not found, skipping.")
            continue
        x_arr = sio.loadmat(x_path)["PHM_testX"].astype(np.float32)
        y_arr = sio.loadmat(y_path)["PHM_testY"].flatten().astype(np.float32)
        print(f"  [test ] {bearing}  {x_arr.shape}")
        result.append((bearing, x_arr, y_arr))
    return result


def plot_training_results(history, bearing_preds, save_dir, condition):
    """
    history       : {"train_loss", "test_loss", "rmse", "mae"}
    bearing_preds : list of (bearing_name, pred_np, true_np)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        print(f"Matplotlib 無法載入，略過訓練圖輸出: {exc}")
        return

    epochs    = range(1, len(history["train_loss"]) + 1)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    n_bearings     = len(bearing_preds)
    COLS           = 3
    bearing_rows   = (n_bearings + COLS - 1) // COLS  # ceil div
    total_rows     = 1 + bearing_rows                  # row-0: metrics; rest: bearings
    fig_h          = 5 * total_rows

    fig = plt.figure(figsize=(16, fig_h))
    fig.suptitle(f"DAST Training Results — PHM2012 (FEMTO) Condition {condition}",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(total_rows, COLS, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Train+Test Loss / Train+Test RMSE / MAE ───────
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, history["train_loss"], color="steelblue",
                 label="Train Loss", linewidth=1.2)
    ax_loss.plot(epochs, history["test_loss"], color="tomato",
                 label="Test Loss",  linewidth=1.2)
    ax_loss.set_title("Train vs Test Loss (RMSE)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("RMSE Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_rmse.plot(epochs, history["rmse"], color="tomato", linewidth=1.2)
    best_epoch = int(np.argmin(history["rmse"])) + 1
    ax_rmse.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7,
                    label=f"Best epoch {best_epoch}")
    ax_rmse.set_title("Test RMSE (HI)")
    ax_rmse.set_xlabel("Epoch")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(True, linestyle="--", alpha=0.5)

    ax_mae = fig.add_subplot(gs[0, 2])
    ax_mae.plot(epochs, history["mae"], color="mediumseagreen")
    ax_mae.set_title("Test MAE (HI)")
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE")
    ax_mae.grid(True, linestyle="--", alpha=0.5)

    # ── Rows 1+: one subplot per bearing ─────────────────────
    for i, (name, pred_np, true_np) in enumerate(bearing_preds):
        row = 1 + i // COLS
        col = i % COLS
        ax = fig.add_subplot(gs[row, col])

        ax.plot(true_np, label="True HI", linewidth=1,   color="steelblue")
        ax.plot(pred_np, label="Pred HI", linewidth=1,   color="tomato",
                alpha=0.85, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Sample", fontsize=7)
        ax.set_ylabel("HI", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=7)

    # hide any unused bearing subplots
    for j in range(n_bearings, bearing_rows * COLS):
        row = 1 + j // COLS
        col = j % COLS
        fig.add_subplot(gs[row, col]).set_visible(False)

    save_path = os.path.join(save_dir, f"training_result_FEMTO_cond{condition}{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"圖表已儲存至: {save_path}")

# ── 損失函數 ──────────────────────────────────────────────────
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

# ── 主流程 ────────────────────────────────────────────────────

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    _ds  = config["femto"]
    _tr  = config["training"]
    _mdl = config["model"]

    set_seed(_tr.get("seed", 42))

    data_root  = _ds["output_path"]
    CONDITION  = _ds["condition"]          # "1" / "2" / "3"
    BATCH_SIZE = _tr["batch_size"]
    EPOCHS     = _tr["epochs"]
    LR         = _tr["learning_rate"]
    model_dir  = _tr["model_save_path"]
    WS=_ds["window_size"]
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}  工況: Condition {CONDITION}\n")

    # ── 資料載入 ──
    print("載入訓練資料...")
    trainX_np, trainY_np = load_femto_split(data_root, CONDITION, is_train=True)
    print(f"\n載入測試資料...")
    test_bearings = load_femto_test_bearings(data_root, CONDITION)   # [(name, X, Y), ...]
    testX_np  = np.concatenate([b[1] for b in test_bearings], axis=0)
    testY_np  = np.concatenate([b[2] for b in test_bearings], axis=0)
    print(f"\nTrain: {trainX_np.shape}  Test: {testX_np.shape}\n")

    trainX = torch.tensor(trainX_np, dtype=torch.float32).to(device)
    testX  = torch.tensor(testX_np,  dtype=torch.float32).to(device)
    trainY = torch.tensor(trainY_np, dtype=torch.float32).to(device)
    testY  = torch.tensor(testY_np,  dtype=torch.float32).to(device)

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
    loss_fn   = RMSELoss
    best_rmse = 1e9
    loss_history = {"train_loss": [], "train_rmse": [], "test_loss": [], "rmse": [], "mae": []}

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
            pred_np   = model(testX).squeeze().cpu().numpy()
            test_loss = float(RMSELoss(
                model(testX).squeeze(), testY
            ))

        train_loss_avg = total_loss / len(train_loader)
        train_rmse     = train_loss_avg
        rmse = float(np.sqrt(np.mean((pred_np - testY_np) ** 2)))
        mae  = float(np.mean(np.abs(pred_np - testY_np)))

        loss_history["train_loss"].append(train_loss_avg)
        loss_history["train_rmse"].append(train_rmse)
        loss_history["test_loss"].append(test_loss)
        loss_history["rmse"].append(rmse)
        loss_history["mae"].append(mae)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss_avg:.5f}  Train RMSE: {train_rmse:.5f} "
              f"| Test Loss: {test_loss:.5f}  Test RMSE: {rmse:.5f} | MAE: {mae:.5f}")

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

    best_ckpt = f"dast_FEMTO_cond{CONDITION}_best.pth"
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    bearing_preds = []
    with torch.no_grad():
        for name, x_np, y_np in test_bearings:
            x_t    = torch.tensor(x_np, dtype=torch.float32).to(device)
            pred_b = model(x_t).squeeze().cpu().numpy()
            bearing_preds.append((name, pred_b, y_np))
    plot_training_results(loss_history, bearing_preds, save_dir="plots", condition=CONDITION)


if __name__ == "__main__":
    main()
