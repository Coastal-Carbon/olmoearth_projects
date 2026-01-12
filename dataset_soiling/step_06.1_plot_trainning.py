"""
Plots training metrics from PyTorch Lightning CSV logs for regression.
Shows loss curves and MSE over epochs.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Set the path to your metrics CSV file
# CSV_PATH = Path("logs/lightning_logs/version_11/metrics.csv")
# CSV_PATH = Path("logs/lightning_logs/version_13/metrics.csv")
# Load multiple metric files and merge them
# CSV_PATHS = [
#     Path("/home/wajahat/github/olmoearth_projects/logs/lightning_logs/version_13/metrics.csv"),
#     Path("/home/wajahat/github/olmoearth_projects/logs/lightning_logs/version_15/metrics.csv")
# ]
CSV_PATHS = [
    Path("/home/wajahat/github/olmoearth_projects/logs_reg/lightning_logs/version_4/metrics.csv"),
]

# Load and concatenate all metrics
dfs = []
for path in CSV_PATHS:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    dfs.append(pd.read_csv(path))

df = pd.concat(dfs, ignore_index=True)

# Sort by epoch to ensure proper ordering
df = df.sort_values('epoch').reset_index(drop=True)


# Basic summaries
print("=" * 60)
print("TRAINING SUMMARY (REGRESSION)")
print("=" * 60)

mse_data = df.dropna(subset=["val_mse"])
train_data = df.dropna(subset=["train_loss"])
val_data = df.dropna(subset=["val_loss"])

print(f"Total epochs trained: {int(mse_data['epoch'].max())}")

best_mse = mse_data["val_mse"].min()
best_mse_epoch = int(mse_data.loc[mse_data["val_mse"].idxmin(), "epoch"])
print(f"Best MSE: {best_mse:.6f} (epoch {best_mse_epoch})")

best_rmse = best_mse ** 0.5
print(f"Best RMSE: {best_rmse:.6f}")

print("\nFinal Epoch Stats:")
print(f"  Train Loss: {train_data['train_loss'].iloc[-1]:.6f}")
print(f"  Val Loss:   {val_data['val_loss'].iloc[-1]:.6f}")
print(f"  Val MSE:    {mse_data['val_mse'].iloc[-1]:.6f}")
print(f"  Val RMSE:   {mse_data['val_mse'].iloc[-1] ** 0.5:.6f}")
print("=" * 60)


# Plot metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))


# Loss curves
axes[0].plot(train_data["epoch"], train_data["train_loss"], label="Train", linewidth=2)
axes[0].plot(val_data["epoch"], val_data["val_loss"], label="Val", linewidth=2)
axes[0].set_title("Training & Validation Loss", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)


# MSE
axes[1].plot(mse_data["epoch"], mse_data["val_mse"], linewidth=2)
axes[1].axhline(
    y=best_mse,
    linestyle="--",
    alpha=0.5,
    label=f"Best: {best_mse:.4f}"
)
axes[1].set_title("Validation MSE", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE")
axes[1].legend()
axes[1].grid(alpha=0.3)


# Save + show figure
versions = "_".join([p.parent.name for p in CSV_PATHS])
output_file = f"training_metrics_{versions}.png"


plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_file}")
plt.show()