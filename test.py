import argparse
import json

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset

from DAST_Network import DAST
from DAST_test import compute_mae, compute_rmse, s_score, squeeze_last_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DAST checkpoint.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to a DAST state_dict checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_config = config["cmapss"]
    training_config = config["training"]
    model_config = config["model"]

    dataset = dataset_config["dataset"]
    data_dir = dataset_config["output_path"]
    rul_max = dataset_config["rul_max"]
    batch_size = training_config["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_x = sio.loadmat(
        f"{data_dir}/{dataset}_window_size_testX_new.mat"
    )["test1X_new"]
    test_y = sio.loadmat(
        f"{data_dir}/{dataset}_window_size_testY.mat"
    )["test1Y"].flatten()

    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_loader = DataLoader(
        TensorDataset(test_x), batch_size=batch_size, shuffle=False
    )

    model = DAST(
        dim_val_s=model_config["dim_val"],
        dim_attn_s=model_config["dim_attn"],
        dim_val_t=model_config["dim_val"],
        dim_attn_t=model_config["dim_attn"],
        dim_val=model_config["dim_val"],
        dim_attn=model_config["dim_attn"],
        time_step=test_x.shape[1],
        input_size=test_x.shape[2],
        dec_seq_len=model_config["dec_seq_len"],
        out_seq_len=model_config["out_seq_len"],
        n_encoder_layers=model_config["n_encoder_layers"],
        n_decoder_layers=model_config["n_decoder_layers"],
        n_heads=model_config["n_heads"],
        dropout=model_config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            prediction = squeeze_last_dim(model(batch_x.to(device)))
            predictions.append(torch.clamp(prediction, min=0).cpu())

    prediction_np = torch.cat(predictions).numpy()
    test_y = np.asarray(test_y)

    rmse = compute_rmse(prediction_np, test_y, rul_max)
    mae = compute_mae(prediction_np, test_y, rul_max)
    score = s_score(test_y * rul_max, prediction_np * rul_max)

    print(f"Device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Test samples: {len(test_y)}")
    print(f"Test RMSE: {rmse:.5f}")
    print(f"Test MAE: {mae:.5f}")
    print(f"Test Score: {score:.5f}")


if __name__ == "__main__":
    main()
