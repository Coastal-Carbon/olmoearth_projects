import rasterio
import numpy as np
from pathlib import Path

BASE_DIR = Path("dataset_pastis/windows/predict")
NUM_CLASSES = 20

def compute_metrics(label, pred, num_classes=20):
    # Valid mask
    valid_mask = (
        (label >= 0) & (label < num_classes) &
        (pred >= 0) & (pred < num_classes)
    )

    # Accuracy
    accuracy = (pred[valid_mask] == label[valid_mask]).sum() / valid_mask.sum()

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(label[valid_mask].flatten(), pred[valid_mask].flatten()):
        cm[t, p] += 1

    # IoUs
    ious = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)

    miou = np.mean(ious) if len(ious) > 0 else 0.0
    return accuracy, miou

def load_geotiff(path):
    with rasterio.open(path) as src:
        return src.read(1)

chips = [d for d in BASE_DIR.iterdir() if d.is_dir()]
chips.sort()

all_acc = []
all_miou = []

print(f"Found {len(chips)} chips\n")

for chip_dir in chips:
    chip_name = chip_dir.name
    base = chip_dir / "layers"

    label_path = base / "label/B1/geotiff.tif"
    pred_path = base / "output/output/geotiff.tif"

    if not label_path.exists() or not pred_path.exists():
        print(f"Skipping {chip_name} (missing files)")
        continue

    label = load_geotiff(label_path)
    pred = load_geotiff(pred_path)

    acc, miou = compute_metrics(label, pred, NUM_CLASSES)

    all_acc.append(acc)
    all_miou.append(miou)

    print(f"{chip_name}:  Accuracy={acc*100:.2f}%   mIoU={miou:.4f}")

# Print dataset-level metrics
mean_acc = np.mean(all_acc) * 100
mean_miou = np.mean(all_miou)

print("\n=====================================")
print(f"Mean Accuracy across all chips: {mean_acc:.2f}%")
print(f"Mean mIoU across all chips:     {mean_miou:.4f}")
print("=====================================")
