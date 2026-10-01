import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans as scipy_kmeans, vq as scipy_vq


def _compute_rmse(pred, true, rul_max):
    return float(np.sqrt(np.mean((pred - true) ** 2)) * rul_max)


def _s_score(y_true, y_pred):
    diff = np.array(y_pred) - np.array(y_true)
    score = [
        (math.exp(-d / 13.0) - 1.0) if d < 0 else (math.exp(d / 10.0) - 1.0)
        for d in diff
    ]
    return float(np.sum(score))


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
            cluster_rmse.append(_compute_rmse(pred_np[mask], testY_np[mask], RUL_max))
            cluster_score.append(_s_score(true_rul[mask], pred_rul[mask]))
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
