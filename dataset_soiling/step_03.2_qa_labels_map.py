"""
Visualize Label TIFs and Bboxes

Creates:
1. Grid plot of sample label TIFs with their normalized values and lat/lon
2. Folium map showing bbox locations with device polygons
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import folium
from pyproj import Transformer
from shapely.geometry import shape

# === CONFIG ===
LABELS_DIR = "dataset_soiling/data/labels"
METADATA_FILE = "dataset_soiling/data/metadata/train_metadata.json"
SOILING_FILE = "dataset_soiling/soiling_with_stac.parquet"
OUTPUT_GRID = "dataset_soiling/label_tifs_grid.png"
OUTPUT_MAP = "dataset_soiling/label_tifs_map.html"

N_GRID = 10   # Number of TIFs to show in grid plot
N_MAP = 50    # Number of entries to show on map


def load_metadata(filepath: str) -> dict:
    """Load train metadata JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_portion_geometries(parquet_path: str) -> dict:
    """Load portion geometries from parquet, keyed by device ID."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    
    # Get unique device -> portion mapping
    device_portions = df.drop_duplicates(subset=["ID_DEVICE"])[["ID_DEVICE", "portion"]]
    return dict(zip(device_portions["ID_DEVICE"], device_portions["portion"]))


def bbox_utm_to_wgs84(bbox: list, utm_epsg: int) -> list:
    """Convert UTM bbox to WGS84 for folium."""
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    
    minx, miny, maxx, maxy = bbox
    
    # Transform all four corners
    sw = transformer.transform(minx, miny)  # southwest
    ne = transformer.transform(maxx, maxy)  # northeast
    
    # Return as [[south, west], [north, east]] for folium
    return [[sw[1], sw[0]], [ne[1], ne[0]]]


def create_grid_plot(entries: list, labels_dir: Path, output_path: str, n_samples: int):
    """Create grid visualization of sample TIFs."""
    # Sample entries
    if len(entries) > n_samples:
        # Sample evenly across the dataset
        indices = np.linspace(0, len(entries) - 1, n_samples, dtype=int)
        sampled_entries = [entries[i] for i in indices]
    else:
        sampled_entries = entries
    
    n = len(sampled_entries)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    fig.suptitle(f"Label TIFs - Normalized Soiling Index (showing {n} of {len(entries)})", fontsize=16, fontweight="bold")
    
    # Flatten axes for easy iteration
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, entry in enumerate(sampled_entries):
        ax = axes[idx]
        filepath = labels_dir / entry["filename"]
        
        if filepath.exists():
            with rasterio.open(filepath) as src:
                data = src.read(1)
                
                im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Title with key info including lat/lon
        title = (
            f"{entry['id_plant'][:20]}\n"
            f"{entry['id_device'][:20]}\n"
            f"{entry['date']}\n"
            f"lat={entry['centroid_lat']:.4f}, lon={entry['centroid_lon']:.4f}\n"
            f"norm={entry['soiling_normalized']:.3f} | raw={entry['soiling_raw']:.3f}"
        )
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    
    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved grid plot ({n} samples): {output_path}")


def create_folium_map(entries: list, portion_geoms: dict, output_path: str, n_samples: int):
    """Create folium map with bbox rectangles and original portion polygons."""
    # Sample entries
    total_entries = len(entries)
    if len(entries) > n_samples:
        # Sample evenly across the dataset
        indices = np.linspace(0, len(entries) - 1, n_samples, dtype=int)
        sampled_entries = [entries[i] for i in indices]
    else:
        sampled_entries = entries
    
    # Calculate center from all centroids
    lats = [e["centroid_lat"] for e in sampled_entries]
    lons = [e["centroid_lon"] for e in sampled_entries]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3)
    
    # Color scale based on normalized value
    def get_color(norm_val):
        if norm_val < 0.3:
            return "red"
        elif norm_val < 0.6:
            return "orange"
        elif norm_val < 0.8:
            return "yellow"
        else:
            return "green"
    
    # Add bbox rectangles and portion polygons
    for entry in sampled_entries:
        bbox_wgs84 = bbox_utm_to_wgs84(entry["bbox"], entry["utm_epsg"])
        color = get_color(entry["soiling_normalized"])
        
        # Popup with info
        popup_html = f"""
        <b>{entry['id_plant']}</b><br>
        Device: {entry['id_device']}<br>
        Date: {entry['date']}<br>
        Cloud: {entry['cloud_cover']:.2f}%<br>
        <hr>
        <b>Centroid:</b><br>
        Lat: {entry['centroid_lat']:.6f}<br>
        Lon: {entry['centroid_lon']:.6f}<br>
        <hr>
        Raw: {entry['soiling_raw']:.4f}<br>
        Clipped: {entry['soiling_clipped']:.4f}<br>
        Normalized: {entry['soiling_normalized']:.4f}<br>
        <hr>
        UTM EPSG: {entry['utm_epsg']}<br>
        """
        
        # Add generated bbox rectangle (colored by soiling value)
        folium.Rectangle(
            bounds=bbox_wgs84,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.3,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"BBOX: {entry['id_device']} ({entry['date']})"
        ).add_to(m)
        
        # Add centroid marker in the middle of bbox
        folium.CircleMarker(
            location=[entry["centroid_lat"], entry["centroid_lon"]],
            radius=4,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"CENTROID: {entry['centroid_lat']:.4f}, {entry['centroid_lon']:.4f}"
        ).add_to(m)
        
        # Add original portion polygon (black outline only)
        device_id = entry["id_device"]
        if device_id in portion_geoms:
            portion_json = json.loads(portion_geoms[device_id])
            
            folium.GeoJson(
                portion_json,
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": "black",
                    "weight": 2,
                    "fillOpacity": 0,
                },
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"PORTION: {device_id}"
            ).add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px;">
        <b>Soiling Index (Normalized)</b><br>
        <i style="background: green; width: 12px; height: 12px; display: inline-block;"></i> 0.8 - 1.0 (Clean)<br>
        <i style="background: yellow; width: 12px; height: 12px; display: inline-block;"></i> 0.6 - 0.8<br>
        <i style="background: orange; width: 12px; height: 12px; display: inline-block;"></i> 0.3 - 0.6<br>
        <i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> 0.0 - 0.3 (Dirty)<br>
        <hr>
        <b>Shapes</b><br>
        <i style="border: 2px solid green; background: rgba(0,128,0,0.3); width: 12px; height: 12px; display: inline-block;"></i> Generated Bbox (32x32 @ 10m)<br>
        <i style="border: 2px solid black; width: 12px; height: 12px; display: inline-block;"></i> Original Portion Polygon<br>
        <i style="background: white; border: 2px solid black; width: 8px; height: 8px; display: inline-block; border-radius: 50%;"></i> Centroid<br>
        <hr>
        <b>Showing {len(sampled_entries)} of {total_entries} entries</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Fit bounds to show all markers
    all_lats = [e["centroid_lat"] for e in sampled_entries]
    all_lons = [e["centroid_lon"] for e in sampled_entries]
    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
    
    m.save(output_path)
    print(f"Saved folium map ({len(sampled_entries)} samples): {output_path}")


def main():
    labels_dir = Path(LABELS_DIR)
    metadata = load_metadata(METADATA_FILE)
    entries = metadata["entries"]
    
    print(f"Total entries: {len(entries)}")
    print(f"Grid samples: {N_GRID}, Map samples: {N_MAP}")
    
    # Load portion geometries for map
    portion_geoms = load_portion_geometries(SOILING_FILE)
    print(f"Loaded {len(portion_geoms)} portion geometries")
    
    create_grid_plot(entries, labels_dir, OUTPUT_GRID, N_GRID)
    create_folium_map(entries, portion_geoms, OUTPUT_MAP, N_MAP)
    
    print("\nDone!")


if __name__ == "__main__":
    main()