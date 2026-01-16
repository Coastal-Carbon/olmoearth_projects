"""
Plot All Unique Plants on Map

Loads merged soiling data and plots unique plant locations
with date range in popup labels.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import folium
from shapely.geometry import shape

# === CONFIG ===
INPUT_FILE = "dataset_soiling/merged_soiling.parquet"
OUTPUT_MAP = "dataset_soiling/all_plants_map.html"


def load_and_aggregate(filepath: str) -> pd.DataFrame:
    """Load data and aggregate to unique plants."""
    print(f"Loading data from {filepath}")
    df = pd.read_parquet(filepath)
    
    # Aggregate per plant
    plant_info = df.groupby("ID_PLANT").agg(
        device_count=("ID_DEVICE", "nunique"),
        row_count=("ID_DEVICE", "count"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        portion_sample=("portion", "first"),  # Get one portion for centroid
    ).reset_index()
    
    print(f"Found {len(plant_info)} unique plants")
    return plant_info


def get_centroid(portion_json: str) -> tuple[float, float]:
    """Get centroid from portion geometry."""
    try:
        geom = shape(json.loads(portion_json))
        return geom.centroid.y, geom.centroid.x  # lat, lon
    except:
        return None, None


def create_map(plant_info: pd.DataFrame, output_path: str):
    """Create folium map with all plants."""
    
    # Calculate centroids
    plant_info["centroid"] = plant_info["portion_sample"].apply(
        lambda x: get_centroid(x) if pd.notna(x) else (None, None)
    )
    plant_info["lat"] = plant_info["centroid"].apply(lambda x: x[0])
    plant_info["lon"] = plant_info["centroid"].apply(lambda x: x[1])
    
    # Filter out plants without valid coordinates
    valid_plants = plant_info[plant_info["lat"].notna()].copy()
    print(f"Plants with valid coordinates: {len(valid_plants)}")
    
    # Calculate map center
    center_lat = valid_plants["lat"].mean()
    center_lon = valid_plants["lon"].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3)
    
    # Add markers for each plant
    for _, row in valid_plants.iterrows():
        date_min = row["date_min"].strftime("%Y-%m-%d")
        date_max = row["date_max"].strftime("%Y-%m-%d")
        
        popup_html = f"""
        <b>{row['ID_PLANT']}</b><br>
        <hr>
        <b>Devices:</b> {row['device_count']}<br>
        <b>Total rows:</b> {row['row_count']:,}<br>
        <hr>
        <b>Date range:</b><br>
        {date_min} to {date_max}<br>
        <hr>
        <b>Location:</b><br>
        Lat: {row['lat']:.6f}<br>
        Lon: {row['lon']:.6f}<br>
        """
        
        tooltip = f"{row['ID_PLANT']} ({date_min} to {date_max})"
        
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color="darkblue",
            fill=True,
            fill_color="steelblue",
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip
        ).add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; z-index: 1000; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px;">
        <b>All Plants</b><br>
        Total: {len(valid_plants)} plants<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Fit bounds
    m.fit_bounds([
        [valid_plants["lat"].min(), valid_plants["lon"].min()],
        [valid_plants["lat"].max(), valid_plants["lon"].max()]
    ])
    
    m.save(output_path)
    print(f"Saved map: {output_path}")


def print_summary(plant_info: pd.DataFrame):
    """Print summary of plants."""
    print("\n" + "=" * 70)
    print("PLANT SUMMARY")
    print("=" * 70)
    print(f"  Total plants:                    {len(plant_info):>10}")
    print(f"  Total devices:                   {plant_info['device_count'].sum():>10}")
    print(f"  Total rows:                      {plant_info['row_count'].sum():>10,}")
    print()
    print("  TOP 10 PLANTS BY DEVICE COUNT:")
    print("-" * 70)
    top10 = plant_info.nlargest(10, "device_count")[["ID_PLANT", "device_count", "row_count", "date_min", "date_max"]]
    for _, row in top10.iterrows():
        date_range = f"{row['date_min'].strftime('%Y-%m-%d')} to {row['date_max'].strftime('%Y-%m-%d')}"
        print(f"  {row['ID_PLANT']:<30} {row['device_count']:>3} devices | {row['row_count']:>6,} rows | {date_range}")
    print("=" * 70)


def main():
    plant_info = load_and_aggregate(INPUT_FILE)
    print_summary(plant_info)
    create_map(plant_info, OUTPUT_MAP)
    print("\nDone!")


if __name__ == "__main__":
    main()