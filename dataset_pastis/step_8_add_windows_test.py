"""
Create test/validation windows from PASTIS-HD by randomly selecting patches.
"""

import json
import random
import subprocess
from pathlib import Path

# ---------------------------------------
n = 10           # number of patches to select
seed = 42
# ---------------------------------------

# Path to metadata
metadata_path = Path("pastis_hd_target/metadata/train_metadata.json")

# RSLEARN output root
root = Path("dataset_pastis")

# Load metadata
with metadata_path.open() as f:
    patches = json.load(f)

# Random sampling
random.seed(seed)
selected_patches = random.sample(patches, n)

print(f"Processing {len(selected_patches)} test windows (seed={seed})")

for idx, p in enumerate(selected_patches, start=1):
    patch_id = p["patch_id"]
    minx, miny, maxx, maxy = p["bbox"]
    crs = p["crs"]
    start = p["date_min"]
    end = p["date_max"]

    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    name = f"pastis_{patch_id}"

    print(f"\n[{idx}/{len(selected_patches)}] {name}")

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
        print("  ✓ SUCCESS")
    else:
        print("  ✗ FAILED")
        print("  STDERR:", result.stderr.strip())