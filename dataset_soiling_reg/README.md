# Solar Panel Soiling Prediction - Scalar Regression

Scalar regression for predicting soiling index from Sentinel-2 imagery using OlmoEarth with RegressionHead.
This is a follow-up to the per-pixel regression experiment, using a simpler scalar target per window to see whether performance improves (or degrades) and to establish a baseline for comparison.

Uses parquet file from `dataset_soiling` as source data.

## Key Files

- `model.yaml` - Model and training config for OlmoEarth encoder + regression head.
- `config.json` - Dataset config generated for this directory (used by rslearn).
- `step_03_prepare.md` - STAC search instructions for Sentinel-2 items.
- `step_04_ingest.md` - Dataset ingest steps for labels and imagery.
- `step_05_materialize.md` - Materialization steps for windows/tiles.
- `data/metadata/normalization.json` - Normalization params from label generation.
- `data/metadata/train_metadata.json` - Label metadata for training/validation.

## Directory Structure

```
dataset_soiling_reg/
├── README.md
├── config.json
├── model.yaml
│
├── step_01_generate_labels.py
├── step_02_add_windows.py
├── step_03_prepare.md
├── step_04_ingest.md
├── step_05_materialize.md
├── step_06_patch_labels.py
├── step_07_split.py
├── step_08.1_qa_check_labels.py
├── step_08.2_qa_check_sen2.py
├── step_08.3_qa_patch_logic.py
├── step_09_train.md
├── step_10_predict.md
├── step_11.1_eval_prediction.py
├── step_11.2_plot_scatter_color.py
├── step_11.3_plot_box.py
├── step_11.4_plot_box_tif.py
├── step_11.5_plot_overtime.py
├── step_11.6_compare_spread.py
├── util_re_normalize.py
│
├── data/
│   ├── labels/
│   │   └── *.geojson
│   └── metadata/
│       ├── train_metadata.json
│       └── normalization.json
│
├── tiles/
│   └── label/
│       └── {label_name}/
│
├── windows/
│   ├── default/
│   │   └── {window_name}/
│   │       ├── items.json
│   │       ├── metadata.json
│   │       └── layers/
│   │           ├── label/
│   │           │   ├── data.geojson
│   │           │   └── completed
│   │           └── sentinel2_l2a/
│   │               ├── B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/
│   │               │   └── geotiff.tif
│   │               └── completed
│   └── predict/
│       └── {window_name}/
│           ├── items.json
│           ├── metadata.json
│           └── layers/
│               ├── label/
│               │   ├── data.geojson
│               │   └── completed
│               ├── sentinel2_l2a/
│               │   ├── B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/
│               │   │   └── geotiff.tif
│               │   └── completed
│               └── output/
│                   ├── data.geojson
│                   └── completed
│
└── cache/
    └── planetary_computer/
        └── *.json

Note: checkpoints_reg/ and logs_reg/ are created at repo root, not under dataset_soiling_reg/
```

## Steps

- `step_01_generate_labels.py` - Generate GeoJSON labels from parquet
- `step_02_add_windows.py` - Create windows with train/test split
- `step_03_prepare.md` - Fetch STAC items from Planetary Computer
- `step_04_ingest.md` - Ingest labels into dataset
- `step_05_materialize.md` - Download and materialize Sentinel-2 imagery
- `step_06_patch_labels.py` - Update labels in materialized windows
- `step_07_split.py` - Split default group into train/val
- `step_08.1_qa_check_labels.py` - Check label consistency
- `step_08.2_qa_check_sen2.py` - Check Sentinel-2 availability
- `step_08.3_qa_patch_logic.py` - Validate patch logic
- `step_09_train.md` - Train model
- `step_10_predict.md` - Run predictions on test set
- `step_11.1_eval_prediction.py` - Compute metrics and plots
- `step_11.2_plot_scatter_color.py` - Scatter plot by device
- `step_11.3_plot_box.py` - Box plots by device
- `step_11.4_plot_box_tif.py` - Compare with per-pixel model
- `step_11.5_plot_overtime.py` - Temporal analysis
- `step_11.6_compare_spread.py` - Compare variance between models
- `util_re_normalize.py` - Convert normalized metrics to original scale
