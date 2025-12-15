"""
Split soiling dataset windows into train/val/test groups.

This script:
1. Splits the 'default' group into train (87.5%) and val (12.5%) using hash-based assignment
2. Clears options from the 'predict' group (used for testing)

Groups after running:
- default group: train/val split via metadata options
- predict group: test set with empty options
"""

from rslearn.dataset import Dataset
from upath import UPath
import hashlib
import tqdm

# Initialize dataset
ds = Dataset(UPath("./dataset_soiling"))

# Split default group into train/val
print("Splitting default group into train/val...")
windows = ds.load_windows(groups=["default"], workers=16)

for w in tqdm.tqdm(windows, desc="Processing default"):
    hexid = hashlib.sha256(w.name.encode()).hexdigest()[0]
    w.options["split"] = "val" if hexid in ["0", "1"] else "train"
    w.save()

# Clear options from predict group (test set)
print("\nClearing options from predict group...")
windows = ds.load_windows(groups=["predict"], workers=16)

for w in tqdm.tqdm(windows, desc="Processing predict"):
    w.options = {}
    w.save()

print("\nComplete!")
print("- default group: split into train/val")
print("- predict group: options cleared (test set)")