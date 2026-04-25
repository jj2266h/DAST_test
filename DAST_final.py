# -*- coding: utf-8 -*-
import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

# ============================================================
# 可調參數區
# ============================================================
DATA_ROOT   = r"CMAPSSData"
DATASET     = "FD002"   # 可選："FD001" / "FD002" / "FD003" / "FD004"

WINDOW_SIZE = 60
EPOCHS      = 5
BATCH_SIZE  = 256
BASE_LR     = 1e-3
EARLY_SWITCH_EPOCH = 20   # 前幾個 epoch 用 SmoothL1，其後改用 s_score_tensor

# DAST 模型維度（論文偏向 dim=64, heads=4, layers=2）
D_MODEL          = 64
N_HEADS          = 4
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 1
DEC_SEQ_LEN      = 1
OUT_SEQ_LEN      = 1
DROPOUT          = 0.2  

# ============================================================
# reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# loss / metrics
# ============================================================
def s_score_tensor(y_true, y_pred):
    diff = y_pred - y_true
    loss = torch.where(
        diff < 0,
        torch.exp(-diff / 13.0) - 1.0,
        torch.exp(diff / 10.0) - 1.0
    )
    return loss.mean()

def s_score(y_true, y_pred):
    diff = np.array(y_pred) - np.array(y_true)
    score = [(math.exp(-d/13.0) - 1.0) if d < 0 else (math.exp(d/10.0) - 1.0) for d in diff]
    return float(np.sum(score))

def rmse_np(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

# ============================================================
# C-MAPSS Loader (FD001 ~ FD004)
# ============================================================
def load_cmapss_fd(root, fd_name):
    train_path = os.path.join(root, f"train_{fd_name}.txt")
    test_path  = os.path.join(root, f"test_{fd_name}.txt")
    rul_path   = os.path.join(root, f"RUL_{fd_name}.txt")

    train = pd.read_csv(train_path, sep=r"\s+", header=None)
    test  = pd.read_csv(test_path,  sep=r"\s+", header=None)
    rul   = pd.read_csv(rul_path,   sep=r"\s+", header=None).values.flatten()

    # 去掉最後兩個空欄
    cols = ['unit', 'cycle'] + [f'op{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
    train = train.iloc[:, :26]
    test  = test.iloc[:, :26]
    train.columns = cols
    test.columns  = cols

    float_cols = [f'op{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
    train[float_cols] = train[float_cols].astype(np.float64)
    test[float_cols]  = test[float_cols].astype(np.float64)

    # RUL (train)
    train['RUL'] = train.groupby('unit')['cycle'].transform(lambda x: x.max() - x)

    # RUL (test)
    max_cycle_test = test.groupby('unit')['cycle'].max().to_dict()
    test_units = sorted(test['unit'].unique())
    rul_map = {u: r for u, r in zip(test_units, rul)}

    def calc_test_rul(row):
        u = row['unit']
        return rul_map[u] + (max_cycle_test[u] - row['cycle'])

    test['RUL'] = test.apply(calc_test_rul, axis=1)

    # 多工況資料集 clip RUL=125（對 FD002 / FD004）
    if fd_name in ("FD002", "FD004"):
        train['RUL'] = train['RUL'].clip(upper=125)
        test['RUL']  = test['RUL'].clip(upper=125)

    return train, test

# ============================================================
# Sensor selection：刪掉 7 個無效感測器
# ============================================================
def build_feature_cols(fd_name=None):
    # 根據論文刪除：s1, s5, s6, s10, s16, s18, s19
    remove = {1, 5, 6, 10, 16, 18, 19}
    sensors = [f"s{i}" for i in range(1, 22) if i not in remove]
    return sensors  # 最後剩 14 個感測器

# ============================================================
# Sliding windows
# ============================================================
def make_windows(df, feature_cols, window_size):
    X, Y = [], []
    for u in sorted(df['unit'].unique()):
        d = df[df['unit'] == u].sort_values('cycle')
        x = d[feature_cols].values.astype(np.float32)
        y = d['RUL'].values.astype(np.float32)

        if len(x) < window_size:
            continue

        for i in range(len(x) - window_size + 1):
            X.append(x[i:i+window_size])
            Y.append(y[i+window_size-1])

    return np.array(X), np.array(Y)

# ============================================================
# Cluster-based scaling (論文風格：多工況用 cluster min-max)
# ============================================================
def build_cluster_scalers(train_df, feature_cols, fd_name):
    train_df = train_df.copy()
    multi_cond = fd_name in ("FD002", "FD004")

    if multi_cond:
        # KMeans 依 unit 的 (op1, op2, op3) 平均做 3 群 clustering
        unit_ops = train_df.groupby('unit')[['op1', 'op2', 'op3']].mean().values
        kmeans = KMeans(n_clusters=3, random_state=42).fit(unit_ops)

        units = sorted(train_df['unit'].unique())
        unit_to_cluster = {u: int(c) for u, c in zip(units, kmeans.labels_)}
        train_df['cluster'] = train_df['unit'].map(unit_to_cluster)
        n_clusters = 3
    else:
        kmeans = None
        train_df['cluster'] = 0
        n_clusters = 1

    scalers = {}
    for c in range(n_clusters):
        sub = train_df[train_df['cluster'] == c][feature_cols]
        mn = sub.min(axis=0).astype(np.float64)
        mx = sub.max(axis=0).astype(np.float64)
        scalers[c] = (mn, mx)

    # 將 scaling 套回 train_df
    train_scaled = train_df.copy()
    for c, (mn, mx) in scalers.items():
        mask = (train_scaled['cluster'] == c)
        denom = (mx - mn).replace(0, 1)
        train_scaled.loc[mask, feature_cols] = (
            (train_scaled.loc[mask, feature_cols] - mn) / denom
        ).astype(np.float32)

    return train_scaled, kmeans, scalers

# ============================================================
# Attention utility blocks
# ============================================================
def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, dim=-1)

def attention(Q, K, V):
    a = a_norm(Q, K)
    return torch.matmul(a, V)

class Value(nn.Module):
    def __init__(self, dim_input, dim_val):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)
    def forward(self, x):
        return self.fc1(x)

class Key(nn.Module):
    def __init__(self, dim_input, dim_attn):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
    def forward(self, x):
        return self.fc1(x)

class Query(nn.Module):
    def __init__(self, dim_input, dim_attn):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
    def forward(self, x):
        return self.fc1(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn):
        super().__init__()
        self.value = Value(dim_val, dim_val)
        self.key   = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    def forward(self, x, kv=None):
        if kv is None:
            kv = x
        return attention(self.query(x), self.key(kv), self.value(kv))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionBlock(dim_val, dim_attn) for _ in range(n_heads)]
        )
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)
    def forward(self, x, kv=None):
        a = [h(x, kv=kv) for h in self.heads]    # list of (B, L, dim_val)
        a = torch.stack(a, dim=-1)              # (B, L, dim_val, n_heads)
        a = a.flatten(start_dim=2)              # (B, L, dim_val * n_heads)
        x = self.fc(a)                          # (B, L, dim_val)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_even = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        div_odd = torch.exp(
            torch.arange(1, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_even)
        pe[:, 1::2] = torch.cos(position * div_odd)
        pe = pe.unsqueeze(0).transpose(0, 1)   # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x

# ============================================================
# Encoder / Decoder Layers with dropout=0.2 (論文風格)
# ============================================================
class Sensors_EncoderLayer(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.2):
        super().__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        # x: (B, n_sensors_like, dim_val)
        a = self.attn(x)
        x = self.norm1(x + self.dropout1(a))

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + self.dropout2(a))

        return x

class Time_step_EncoderLayer(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.2):
        super().__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        # x: (B, time_step, dim_val)
        a = self.attn(x)
        x = self.norm1(x + self.dropout1(a))

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + self.dropout2(a))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.2):
        super().__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        # x: (B, L_dec, dim_val), enc: (B, L_enc, dim_val)
        a = self.attn1(x)
        x = self.norm1(x + self.dropout1(a))

        a = self.attn2(x, kv=enc)
        x = self.norm2(x + self.dropout2(a))

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + self.dropout3(a))

        return x

# ============================================================
# DAST Model (論文 7F 擴展 + 3 sensor branches + time branch + decoder)
# ============================================================
class DAST(nn.Module):
    def __init__(
        self,
        dim_val_s,
        dim_attn_s,
        dim_val_t,
        dim_attn_t,
        dim_val,
        dim_attn,
        time_step,
        feature_len,     # F = len(feature_cols)
        dec_seq_len,
        out_seq_len,
        n_decoder_layers=1,
        n_encoder_layers=2,
        n_heads=1,
        dropout=0.2,
    ):
        super().__init__()

        # 確保三個分支維度一致
        assert dim_val_s == dim_val_t == dim_val
        assert dim_attn_s == dim_attn_t == dim_attn

        self.dec_seq_len = dec_seq_len
        self.F_base = feature_len          # F
        self.F2     = self.F_base * 2      # 2F
        self.F4     = self.F_base * 4      # 4F
        self.total_feat = self.F_base + self.F2 + self.F4   # 7F

        # F_raw → 7F
        self.expand_fc = nn.Linear(self.F_base, self.total_feat)

        # 三個 sensor encoder branch
        self.sensor_encoder1 = nn.ModuleList([
            Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.sensor_encoder2 = nn.ModuleList([
            Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.sensor_encoder3 = nn.ModuleList([
            Sensors_EncoderLayer(dim_val_s, dim_attn_s, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])

        # 時間編碼分支
        self.time_encoder = nn.ModuleList([
            Time_step_EncoderLayer(dim_val_t, dim_attn_t, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.pos_t = PositionalEncoding(dim_val_t)
        self.timestep_enc_input_fc = nn.Linear(self.total_feat, dim_val_t)

        # sensor path input：最後一維 time_step → dim_val_s
        self.sensor_enc_input_fc = nn.Linear(time_step, dim_val_s)

        # 三個 decoder branch
        self.decoder1 = nn.ModuleList([
            DecoderLayer(dim_val, dim_attn, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder2 = nn.ModuleList([
            DecoderLayer(dim_val, dim_attn, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder3 = nn.ModuleList([
            DecoderLayer(dim_val, dim_attn, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])

        # dec input：F / 2F / 4F → dim_val
        self.dec_input_fc1 = nn.Linear(self.F_base, dim_val)
        self.dec_input_fc2 = nn.Linear(self.F2,     dim_val)
        self.dec_input_fc3 = nn.Linear(self.F4,     dim_val)

        self.norm   = nn.LayerNorm(dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val * 3, out_seq_len)

    def forward(self, x):
        """
        x: (B, T, F_raw), F_raw = feature_len (14 sensors)
        """
        B, T, F_raw = x.shape

        # Step1: F_raw → 7F
        if F_raw != self.total_feat:
            x = self.expand_fc(x)   # (B, T, 7F)

        # Step2: split 7F → (F, 2F, 4F)
        feature1 = x[:, :, :self.F_base]                             # (B, T, F)
        feature2 = x[:, :, self.F_base:self.F_base+self.F2]          # (B, T, 2F)
        feature3 = x[:, :, self.F_base+self.F2:self.total_feat]      # (B, T, 4F)

        # Step3: Sensor path (B, #feature, T) → Linear(time_step → dim_val_s)
        sensor1 = feature1.transpose(1, 2).contiguous()  # (B, F, T)
        sensor2 = feature2.transpose(1, 2).contiguous()  # (B, 2F, T)
        sensor3 = feature3.transpose(1, 2).contiguous()  # (B, 4F, T)

        sensor1 = self.sensor_enc_input_fc(sensor1)  # (B, F,  dim)
        sensor2 = self.sensor_enc_input_fc(sensor2)  # (B, 2F, dim)
        sensor3 = self.sensor_enc_input_fc(sensor3)  # (B, 4F, dim)

        e1 = sensor1
        e2 = sensor2
        e3 = sensor3

        # Step4: Time encoder branch (用 7F 融合特徵)
        t_in = self.timestep_enc_input_fc(x)   # (B, T, dim_val_t)
        t_in = self.pos_t(t_in)
        o = t_in

        for layer in self.sensor_encoder1:
            e1 = layer(e1)
        for layer in self.sensor_encoder2:
            e2 = layer(e2)
        for layer in self.sensor_encoder3:
            e3 = layer(e3)
        for layer in self.time_encoder:
            o = layer(o)

        # Step5: feature fusion
        p1 = self.norm(torch.cat((e1, o), dim=1))   # (B, F + T, dim)
        p2 = self.norm(torch.cat((e2, o), dim=1))
        p3 = self.norm(torch.cat((e3, o), dim=1))

        # Step6: decoder queries：使用最後 dec_seq_len 個 time step 的特徵
        q1 = feature1[:, -self.dec_seq_len:, :]   # (B, dec_len, F)
        q2 = feature2[:, -self.dec_seq_len:, :]   # (B, dec_len, 2F)
        q3 = feature3[:, -self.dec_seq_len:, :]   # (B, dec_len, 4F)

        q1 = self.dec_input_fc1(q1)   # (B, dec_len, dim)
        q2 = self.dec_input_fc2(q2)
        q3 = self.dec_input_fc3(q3)

        for layer in self.decoder1:
            q1 = layer(q1, p1)
        for layer in self.decoder2:
            q2 = layer(q2, p2)
        for layer in self.decoder3:
            q3 = layer(q3, p3)

        # Step7: concat 三個 decoder 輸出
        d = torch.cat((q1, q2, q3), dim=2)  # (B, dec_len, dim*3)

        # Step8：flatten + out
        d_flat = d.flatten(start_dim=1)     # (B, dec_len * dim * 3)
        out = self.out_fc(F.elu(d_flat))    # (B, out_seq_len)
        return out.squeeze(-1)              # (B,)

# ============================================================
# Evaluate (test；使用 cluster-based scaling)
# ============================================================
def evaluate(model, test_df, feature_cols, window_size,
             fd_name, kmeans, scalers, device):

    preds, trues = [], []
    model.eval()
    multi_cond = fd_name in ("FD002", "FD004")

    with torch.no_grad():
        for u in sorted(test_df['unit'].unique()):
            d = test_df[test_df['unit'] == u].sort_values('cycle').copy()

            if multi_cond:
                op_mean = d[['op1', 'op2', 'op3']].mean().values.reshape(1, -1)
                c = int(kmeans.predict(op_mean)[0])
            else:
                c = 0

            mn, mx = scalers[c]
            denom = (mx - mn).replace(0, 1)

            # ⚠ 保持 x_raw 為 DataFrame，讓與 mn/denom 的廣播完全正確
            x_raw = d[feature_cols].astype(np.float64)
            x_scaled_df = ((x_raw - mn) / denom).astype(np.float32)
            x_scaled = x_scaled_df.values

            if len(x_scaled) < window_size:
                x_scaled = np.pad(
                    x_scaled,
                    ((window_size - len(x_scaled), 0), (0, 0)),
                    mode='edge'
                )

            windows = np.stack(
                [x_scaled[i:i+window_size]
                 for i in range(len(x_scaled) - window_size + 1)],
                axis=0
            ).astype(np.float32)

            X = torch.tensor(windows, dtype=torch.float32, device=device)
            out = model(X).cpu().numpy()

            preds.append(float(out[-1]))
            trues.append(float(d['RUL'].values[-1]))

    preds = np.array(preds)
    trues = np.array(trues)

    return rmse_np(trues, preds), mae_np(trues, preds), s_score(trues, preds), preds, trues

# ============================================================
# Main (完整訓練流程)
# ============================================================
def main():
    root = DATA_ROOT
    fd   = DATASET
    W    = WINDOW_SIZE

    print(f"=== 真正論文版 DAST + dropout=0.2 ({fd}) ===")

    # 1. Load data
    train_df, test_df = load_cmapss_fd(root, fd)

    # 2. Sensor selection
    feature_cols = build_feature_cols(fd)

    # 3. Cluster-based scaling
    train_scaled, kmeans, scalers = build_cluster_scalers(
        train_df, feature_cols, fd_name=fd
    )

    # 4. Sliding windows
    X_train, y_train = make_windows(train_scaled, feature_cols, W)
    print("Train windows:", X_train.shape)

    # 5. Model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DAST(
        dim_val_s=D_MODEL, dim_attn_s=D_MODEL,
        dim_val_t=D_MODEL, dim_attn_t=D_MODEL,
        dim_val=D_MODEL,  dim_attn=D_MODEL,
        time_step=W,
        feature_len=len(feature_cols),
        dec_seq_len=DEC_SEQ_LEN,
        out_seq_len=OUT_SEQ_LEN,
        n_decoder_layers=N_DECODER_LAYERS,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

    best_rmse = 1e9
    best_path = os.path.join(root, f"best_DAST_{fd}_PAPER_DROPOUT.pth")

    # 6. Training loop
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optim.zero_grad()

            out = model(Xb)

            # 前幾個 epoch 用 SmoothL1，之後改用 s_score_tensor（論文常見做法之一）
            if ep < EARLY_SWITCH_EPOCH:
                loss = nn.SmoothL1Loss()(out, yb)
            else:
                loss = s_score_tensor(yb, out)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item() * Xb.size(0)

        total_loss /= len(loader.dataset)

        # Evaluate on test set
        rmse, mae, score, preds, trues = evaluate(
            model, test_df, feature_cols, W,
            fd_name=fd, kmeans=kmeans, scalers=scalers, device=device
        )

        scheduler.step(rmse)

        print(
            f"Epoch {ep:03d}/{EPOCHS} | "
            f"TrainLoss {total_loss:.4f} | "
            f"RMSE {rmse:.4f} | MAE {mae:.4f} | "
            f"s_score {score:.2f} | "
            f"lr {optim.param_groups[0]['lr']:.1e}"
        )

        # 保存最佳模型
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), best_path)

    print("BEST RMSE:", best_rmse)


if __name__ == "__main__":
    main()