import rasterio

# Raw label
with rasterio.open("dataset_soiling/data/labels/CAMPOS_DEL_SOL_DIQ_6_CAMPOS_I_CT29_2024-06-06_label.tif") as src:
    print("RAW LABEL:")
    print(f"  Shape: {src.height} x {src.width}")
    print(f"  Bounds: {src.bounds}")
    print(f"  CRS: {src.crs}")
    print(f"  Transform: {src.transform}")

# RSLearn label
with rasterio.open("dataset_soiling/windows/default/CAMPOS_DEL_SOL_DIQ_6_CAMPOS_I_CT29_2024-06-06/layers/label/B1/geotiff.tif") as src:
    print("\nRSLEARN LABEL:")
    print(f"  Shape: {src.height} x {src.width}")
    print(f"  Bounds: {src.bounds}")
    print(f"  CRS: {src.crs}")
    print(f"  Transform: {src.transform}")

# Window metadata
import json
with open("dataset_soiling/windows/default/CAMPOS_DEL_SOL_DIQ_6_CAMPOS_I_CT29_2024-06-06/metadata.json") as f:
    meta = json.load(f)
    print("\nWINDOW METADATA:")
    print(json.dumps(meta, indent=2))