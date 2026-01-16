# Solar Panel Soiling Prediction - Per-Pixel Regression

Per-pixel regression for predicting soiling index from Sentinel-2 imagery using OlmoEarth.
This is a prototype to see whether OlmoEarth's regression head can learn to predict panel soiling from Sentinel-2 while reusing its existing data pipeline. We start with device-level soiling measurements and portion geometries in the two Excel files; `step_01_merge_excel_files.py` merges them into `merged_soiling.parquet`, and `step_02_search_stac.py` queries the Planetary Computer STAC API for matching Sentinel-2 items per device/date to produce `dataset_soiling/soiling_with_stac.parquet` (with item IDs, cloud cover, and UTM info). Labels are then generated as constant-valued chips centered on each device footprint, and the model pairs an OlmoEarth encoder with a UNet decoder and a regression head so we can quickly validate the approach before investing in a more complex setup.

## Required Input Files

- `dataset_soiling/portion_json (1).xlsx`
- `dataset_soiling/soiling_data (1).xlsx`

## Steps

- `step_00_qa_check_excel_files.py` - Inspect raw Excel files for schema consistency
- `step_01_merge_excel_files.py` - Merge soiling Excel sources
- `step_01.1_qa_map_all_locations.py` - Map all plant/device locations
- `step_02_search_stac.py` - Search STAC for Sentinel-2 matches
- `step_02.1_qa_check_index_range.py` - Validate soiling index range
- `step_02.2_qa_check_clouds.py` - Inspect cloud cover distribution
- `step_02.3_qa_check_utm.py` - Validate UTM assignments
- `step_02.4_qa_portion_area.py` - Check portion area sanity
- `step_03_generate_label.py` - Generate label GeoJSONs
- `step_03.1_qa_labels.py` - QA label contents
- `step_03.2_qa_labels_map.py` - Map labels for visual QA
- `step_03_prepare.md` - Fetch STAC items from Planetary Computer
- `step_04_ingest.md` - Ingest labels into dataset
- `step_05_materialize.md` - Download and materialize Sentinel-2 imagery
- `step_04.1_patch_labels.py` - Patch/repair labels after data prep
- `step_04_add_windows_train_test.py` - Create windows with train/test split
- `step_05_split.py` - Split default group into train/val
- `step_06_train.md` - Train model
- `step_06.1_plot_trainning.py` - Plot training metrics
- `step_06.2_qa_labels_post_fix.py` - QA labels after patching
- `step_07_predict.md` - Run predictions on test set
- `step_07.1_plot_prediction.py` - Plot prediction outputs
- `step_07.2_check_prediction.py` - Check prediction values
- `step_07.3_qa_check_labels.py` - Validate labels vs windows
- `step_08.1_plot_overtime.py` - Temporal analysis plots
- `step_08.2_plot_scatter_color.py` - Scatter plots by device
