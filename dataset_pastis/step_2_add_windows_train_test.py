"""
Generate RSLEARN windows from PASTIS-HD train_metadata.json with train/test split.

This script:
- Loads train_metadata.json
- Selects n patches (or all)
- Splits into 80% training (group=default) and 20% testing (group=predict)
- Calls `rslearn dataset add_windows` for each
- Patches metadata.json for training patches (add split=default)
"""

import json
import random
import subprocess
from pathlib import Path

# Configuration
n = None          # subset size, or set to None for all
seed = 42
split_ratio = 0.8  # 80% train, 20% test

# Path to metadata
metadata_path = Path("pastis_hd_target/metadata/train_metadata.json")

# RSLEARN output root
root = Path("dataset_pastis")

# Load metadata
with metadata_path.open() as f:
    patches = json.load(f)

# Optional sampling
if n is not None:
    random.seed(seed)
    patches = random.sample(patches, n)
    print(f"Selected {len(patches)} patches (seed={seed})")
else:
    print(f"Using all {len(patches)} patches")

# Split into train/test
random.seed(seed)
random.shuffle(patches)

split_idx = int(len(patches) * split_ratio)
train_patches = patches[:split_idx]
test_patches = patches[split_idx:]

print(f"\nTrain patches: {len(train_patches)}")
print(f"Test patches: {len(test_patches)}")
print(f"Total: {len(patches)}")

# Process training patches
print("\nProcessing training patches (group=default)")

for idx, p in enumerate(train_patches, start=1):
    patch_id = p["patch_id"]
    minx, miny, maxx, maxy = p["bbox"]
    crs = p["crs"]
    start = p["date_min"]
    end = p["date_max"]

    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    name = f"pastis_{patch_id}"

    print(f"\n[{idx}/{len(train_patches)}] {name}")

    cmd = [
        "rslearn", "dataset", "add_windows",
        f"--root={root}",
        "--group=default",
        f"--name={name}",
        f"--box={bbox_str}",
        f"--start={start}",
        f"--end={end}",
        f"--src_crs={crs}",
        "--resolution=10",
        "--utm"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("  SUCCESS")
    else:
        print("  FAILED")
        print("  STDERR:", result.stderr.strip())
        continue

    # Patch generated metadata.json
    meta_file = root / "windows" / "default" / name / "metadata.json"

    if meta_file.exists():
        with meta_file.open("r") as f:
            meta = json.load(f)

        if "options" not in meta:
            meta["options"] = {}

        meta["options"]["split"] = "default"

        with meta_file.open("w") as f:
            json.dump(meta, f)

        print("  Patched metadata.json with split=default")
    else:
        print("  WARNING: metadata.json not found for", name)

# Process test patches
print("\nProcessing test patches (group=predict)")

for idx, p in enumerate(test_patches, start=1):
    patch_id = p["patch_id"]
    minx, miny, maxx, maxy = p["bbox"]
    crs = p["crs"]
    start = p["date_min"]
    end = p["date_max"]

    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    name = f"pastis_{patch_id}"

    print(f"\n[{idx}/{len(test_patches)}] {name}")

    cmd = [
        "rslearn", "dataset", "add_windows",
        f"--root={root}",
        "--group=predict",
        f"--name={name}",
        f"--box={bbox_str}",
        f"--start={start}",
        f"--end={end}",
        f"--src_crs={crs}",
        "--resolution=10",
        "--utm"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("  SUCCESS")
    else:
        print("  FAILED")
        print("  STDERR:", result.stderr.strip())

# Summary
print("\nComplete")
print(f"Training patches: {len(train_patches)} (group=default)")
print(f"Test patches: {len(test_patches)} (group=predict)")