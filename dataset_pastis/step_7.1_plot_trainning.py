import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('logs/lightning_logs/version_6/metrics.csv')
df = pd.read_csv('/home/wajahat/github/olmoearth_projects/logs/lightning_logs/version_6/metrics.csv')

# Print summary statistics
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
miou_data = df.dropna(subset=['val_mean_iou'])
acc_data = df.dropna(subset=['val_accuracy'])
train_data = df.dropna(subset=['train_loss'])
val_data = df.dropna(subset=['val_loss'])

print(f"Total epochs trained: {int(miou_data['epoch'].max())}")
print(f"\nBest mIoU: {miou_data['val_mean_iou'].max():.4f} at epoch {int(miou_data.loc[miou_data['val_mean_iou'].idxmax(), 'epoch'])}")
print(f"Best Accuracy: {acc_data['val_accuracy'].max():.4f} at epoch {int(acc_data.loc[acc_data['val_accuracy'].idxmax(), 'epoch'])}")
print(f"\nFinal Epoch Stats:")
print(f"  Train Loss: {train_data['train_loss'].iloc[-1]:.4f}")
print(f"  Val Loss: {val_data['val_loss'].iloc[-1]:.4f}")
print(f"  Val mIoU: {miou_data['val_mean_iou'].iloc[-1]:.4f}")
print(f"  Val Accuracy: {acc_data['val_accuracy'].iloc[-1]:.4f}")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss
axes[0].plot(train_data['epoch'], train_data['train_loss'], label='Train', linewidth=2)
axes[0].plot(val_data['epoch'], val_data['val_loss'], label='Val', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(acc_data['epoch'], acc_data['val_accuracy'], linewidth=2, color='orange')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Val Accuracy', fontsize=11)
axes[1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# mIoU
axes[2].plot(miou_data['epoch'], miou_data['val_mean_iou'], linewidth=2, color='green')
axes[2].axhline(y=miou_data['val_mean_iou'].max(), color='r', linestyle='--', alpha=0.5, label=f'Peak: {miou_data["val_mean_iou"].max():.3f}')
axes[2].set_xlabel('Epoch', fontsize=11)
axes[2].set_ylabel('Val mIoU', fontsize=11)
axes[2].set_title('Validation Mean IoU', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved to training_metrics.png")
plt.show()