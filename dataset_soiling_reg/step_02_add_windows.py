"""
Generate RSLEARN windows from soiling train_metadata.json with train/test split.

This script:
- Loads train_metadata.json (from regression label generation)
- Filters by cloud cover threshold
- Optionally samples N entries with balanced distribution across soiling index
- Splits into 80% training (group=default) and 20% testing (group=predict)
- Calls `rslearn dataset add_windows` for each (parallel)
- Patches metadata.json for training patches (add split=default)
"""

import json
import random
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
N = None                      # Number of samples (None = all)
SEED = 42
SPLIT_RATIO = 0.8               # 80% train, 20% test
CLOUD_COVER_THRESHOLD = 1     # Only entries with cloud_cover < this value
BALANCED_SAMPLING = True        # Use stratified sampling by soiling index
N_BINS = 5                      # Number of bins for balanced sampling
MAX_WORKERS = 32                # Parallel workers

METADATA_PATH = Path("dataset_soiling_reg/data/metadata/train_metadata.json")
ROOT = Path("dataset_soiling_reg")


def load_and_filter_metadata(filepath: Path, cloud_threshold: float) -> list[dict]:
    """Load metadata and filter by cloud cover."""
    logger.info(f"Loading metadata from {filepath}")
    
    with filepath.open() as f:
        data = json.load(f)
    
    entries = data["entries"]
    logger.info(f"Total entries: {len(entries)}")
    
    filtered = [e for e in entries if e["cloud_cover"] < cloud_threshold]
    logger.info(f"After cloud filter (<{cloud_threshold}%): {len(filtered)}")
    
    return filtered


def balanced_sample(entries: list[dict], n: int, n_bins: int, seed: int) -> list[dict]:
    """Sample entries with balanced distribution across soiling index bins."""
    random.seed(seed)
    np.random.seed(seed)
    
    df = pd.DataFrame(entries)
    df["bin"] = pd.cut(df["soiling_normalized"], bins=n_bins, labels=False)
    
    samples_per_bin = n // n_bins
    remainder = n % n_bins
    
    logger.info(f"Balanced sampling: {n} samples across {n_bins} bins ({samples_per_bin} per bin)")
    
    sampled = []
    bin_counts = []
    
    for bin_idx in range(n_bins):
        bin_entries = df[df["bin"] == bin_idx].to_dict("records")
        target = samples_per_bin + (1 if bin_idx < remainder else 0)
        
        if len(bin_entries) <= target:
            sampled.extend(bin_entries)
            bin_counts.append(len(bin_entries))
        else:
            sampled.extend(random.sample(bin_entries, target))
            bin_counts.append(target)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i, count in enumerate(bin_counts):
        logger.info(f"  Bin {i} [{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}]: {count} samples")
    
    random.shuffle(sampled)
    return sampled


def random_sample(entries: list[dict], n: int, seed: int) -> list[dict]:
    """Simple random sampling."""
    random.seed(seed)
    sampled = random.sample(entries, min(n, len(entries)))
    logger.info(f"Random sampling: {len(sampled)} samples")
    return sampled


def split_train_test(entries: list[dict], ratio: float, seed: int) -> tuple[list, list]:
    """Split entries into train and test sets."""
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def get_window_name(entry: dict) -> str:
    """Generate window name from entry filename."""
    return entry["filename"].replace(".geojson", "")


def get_date_range(date_str: str) -> tuple[str, str]:
    """Return start date and end date (start + 1 day)."""
    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=1)
    return date_str, end_date.strftime("%Y-%m-%d")


def process_single_entry(entry: dict, root: Path, group: str, split: str) -> tuple[str, bool]:
    """Process a single entry: add window and patch metadata."""
    name = get_window_name(entry)
    bbox = entry["bbox"]
    utm_epsg = entry["utm_epsg"]
    start_date, end_date = get_date_range(entry["date"])
    
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    crs = f"EPSG:{utm_epsg}"
    
    cmd = [
        "rslearn", "dataset", "add_windows",
        f"--root={root}",
        f"--group={group}",
        f"--name={name}",
        f"--box={bbox_str}",
        f"--start={start_date}",
        f"--end={end_date}",
        f"--src_crs={crs}",
        "--resolution=10",
        "--utm"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return name, False
    
    # Patch metadata if split is provided
    if split:
        meta_file = root / "windows" / group / name / "metadata.json"
        if meta_file.exists():
            with meta_file.open("r") as f:
                meta = json.load(f)
            if "options" not in meta:
                meta["options"] = {}
            meta["options"]["split"] = split
            with meta_file.open("w") as f:
                json.dump(meta, f)
    
    return name, True


def process_entries_parallel(entries: list[dict], root: Path, group: str, split: str, max_workers: int) -> tuple[int, int]:
    """Process entries in parallel."""
    success = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_entry, entry, root, group, split): entry
            for entry in entries
        }
        
        for future in tqdm(as_completed(futures), total=len(entries), desc=f"Creating {group} windows", unit="window"):
            name, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
    
    return success, failed


def print_summary(train: list, test: list, train_ok: int, test_ok: int):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Train windows: {train_ok}/{len(train)}")
    print(f"  Test windows:  {test_ok}/{len(test)}")
    print(f"  Total:         {train_ok + test_ok}/{len(train) + len(test)}")
    
    all_entries = train + test
    soiling_values = [e["soiling_normalized"] for e in all_entries]
    print()
    print(f"  Soiling range: [{min(soiling_values):.3f}, {max(soiling_values):.3f}]")
    print(f"  Soiling mean:  {np.mean(soiling_values):.3f}")
    print(f"  Soiling std:   {np.std(soiling_values):.3f}")
    print("=" * 60)


def main():
    logger.info("Starting RSLEARN window generation")
    logger.info(f"Using {MAX_WORKERS} parallel workers")
    
    entries = load_and_filter_metadata(METADATA_PATH, CLOUD_COVER_THRESHOLD)
    
    if len(entries) == 0:
        logger.error("No entries after filtering")
        return
    
    if N is not None:
        if BALANCED_SAMPLING:
            entries = balanced_sample(entries, N, N_BINS, SEED)
        else:
            entries = random_sample(entries, N, SEED)
    
    train_entries, test_entries = split_train_test(entries, SPLIT_RATIO, SEED)
    
    logger.info(f"Train: {len(train_entries)}, Test: {len(test_entries)}")
    
    train_ok, _ = process_entries_parallel(train_entries, ROOT, "default", "default", MAX_WORKERS)
    test_ok, _ = process_entries_parallel(test_entries, ROOT, "predict", None, MAX_WORKERS)
    
    print_summary(train_entries, test_entries, train_ok, test_ok)
    
    logger.info("Complete")


if __name__ == "__main__":
    main()