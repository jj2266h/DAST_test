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
     
def main():
    # ── 超參數 ──────────────────────────────────────────────
    DATASET    = 'FD004'       # 改成 FD002 / FD003 / FD004
    BATCH_SIZE = 256
    EPOCHS     = 100
    LR         = 1e-3
    RUL_max    = 125.0
    
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    train_dataset='train_dataset'
    model_path = 'train_model'
    
    # ── 資料載入與處理 ─────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainX = load_array(f"{train_dataset}/{DATASET}_window_size_trainX.mat", "train1X")
    trainY = load_array(f"{train_dataset}/{DATASET}_window_size_trainY.mat", "train1Y").flatten()
    testX = load_array(f"{train_dataset}/{DATASET}_window_size_testX.mat", "test1X")
    testY = load_array(f"{train_dataset}/{DATASET}_window_size_testY.mat", "test1Y").flatten()

    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    trainY = torch.tensor(trainY, dtype=torch.float32).to(device)
    testY_np = testY.cpu().numpy() if isinstance(testY, torch.Tensor) else testY  

    train_dataset = TensorDataset(trainX, trainY)
    test_dataset = TensorDataset(testX, testY)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    time_step = train_loader.dataset[0][0].shape[0]  # 獲取時間步長
    input_size = train_loader.dataset[0][0].shape[1]  # 獲取輸入特徵數
    print(f"Time step: {time_step}, Sensor count: {input_size}")
    # ── 模型建立 ─────────────────────────────────────────────
    
    model = DAST(
        dim_val_s=64,
        dim_attn_s=64,
        dim_val_t=64,
        dim_attn_t=64,
        dim_val=64,
        dim_attn=64,
        time_step=time_step,
        input_size=input_size,
        dec_seq_len=10,
        out_seq_len=1,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_heads=4,
        dropout=0.2,
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
        print(f"Epoch {epoch:3d} | Train Loss: {total_loss/len(train_loader):.4f} "
                f"| Test RMSE: {rmse:.4f} | Score: {score_value:.1f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f'dast_{DATASET}_best.pth')

    # ── 儲存模型 ────────────────────────────────────────────
    torch.save(model.state_dict(), f'{model_path}/dast_{DATASET}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth')
    model.to(device)
    print("模型已儲存")

if __name__ == '__main__':
    main()

# python data_process.py