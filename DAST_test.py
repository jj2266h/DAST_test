import numpy as np
import scipy.io as sio
import torch
import os
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
import csv
import json
from datetime import datetime
from DAST_Network import DAST
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_array(path, key):
    return sio.loadmat(path)[key]

def append_experiment_log(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def calculate_score(pred, true):
    diff = pred - true
    return float(np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1)))

def plot_training_results(history, pred_np, testY_np, RUL_max, dataset, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"DAST Training Results — {dataset}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Train Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], marker="o", color="steelblue")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["rmse"], marker="s", color="tomato")
    best_epoch = int(np.argmin(history["rmse"])) + 1
    ax2.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax2.set_title("Test RMSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE (cycles)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Score
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history["score"], marker="^", color="mediumseagreen")
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

    # Pred vs True line (sorted by true RUL)
    ax5 = fig.add_subplot(gs[1, 2])
    sort_idx = np.argsort(testY_np)
    ax5.plot(testY_np[sort_idx] * RUL_max, label="True RUL", linewidth=1)
    ax5.plot(pred_np[sort_idx] * RUL_max, label="Pred RUL", linewidth=1, alpha=0.8)
    ax5.set_title("Sorted RUL Comparison")
    ax5.set_xlabel("Sample (sorted)")
    ax5.set_ylabel("RUL (cycles)")
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.5)

    save_path = os.path.join(save_dir, f"training_result_{dataset}_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"圖表已儲存至: {save_path}")
     
def main():
    # ── 讀取設定檔 ────────────────────────────────────────
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    _ds  = config["cmapss"]
    _tr  = config["training"]
    _mdl = config["model"]

    DATASET       = _ds["dataset"]
    RUL_max       = _ds["rul_max"]
    traindata_dir = _ds["output_path"]
    BATCH_SIZE    = _tr["batch_size"]
    EPOCHS        = _tr["epochs"]
    LR            = _tr["learning_rate"]
    model_dir     = _tr["model_save_path"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # ── 資料載入與處理 ─────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainX = load_array(f"{traindata_dir}/{DATASET}_window_size_trainX_new.mat", "train1X_new")
    trainY = load_array(f"{traindata_dir}/{DATASET}_window_size_trainY.mat", "train1Y").flatten()
    testX = load_array(f"{traindata_dir}/{DATASET}_window_size_testX_new.mat", "test1X_new")
    testY = load_array(f"{traindata_dir}/{DATASET}_window_size_testY.mat", "test1Y").flatten()

    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    trainY = torch.tensor(trainY, dtype=torch.float32).to(device)
    testY_np = testY.cpu().numpy() if isinstance(testY, torch.Tensor) else testY
    testY = torch.tensor(testY, dtype=torch.float32)

    train_dataset = TensorDataset(trainX, trainY)
    test_dataset = TensorDataset(testX, testY)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    time_step = train_loader.dataset[0][0].shape[0]  # 獲取時間步長
    input_size = train_loader.dataset[0][0].shape[1]  # 獲取輸入特徵數
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
    loss_fn = torch.nn.MSELoss()
    best_rmse=2000
    loss_history = {"train_loss": [], "test_loss": [], "rmse": [], "score": []}
    # ── 訓練迴圈 ────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        
        # ── 評估 ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_np = model(testX.to(device)).squeeze().cpu().numpy()
        rmse  = np.sqrt(np.mean((pred_np - testY_np) ** 2))*RUL_max
        score_value = calculate_score(pred_np*RUL_max, testY_np*RUL_max)
        train_loss_avg = total_loss / len(train_loader)
        loss_history["train_loss"].append(train_loss_avg)
        loss_history["rmse"].append(rmse)
        loss_history["score"].append(score_value)
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss_avg:.4f} "
                f"| Test RMSE: {rmse:.4f} | Score: {score_value:.1f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f'dast_{DATASET}_best.pth')

    # ── 儲存模型 ────────────────────────────────────────────
    torch.save(model.state_dict(), f'{model_dir}/dast_{DATASET}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth')
    model.to(device)
    print("模型已儲存")

    # ── 視覺化 ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_pred_np = model(testX.to(device)).squeeze().cpu().numpy()
    plot_training_results(loss_history, final_pred_np, testY_np, RUL_max, DATASET, save_dir="plots")

if __name__ == '__main__':
    main()

# python data_process.py