"""
Validate PASTIS dataset temporal consistency.

Checks that all Sentinel-1 and Sentinel-2 items in items.json
fall within the time_range specified in metadata.json for each patch.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_datetime(dt_string: str) -> datetime:
    if dt_string.endswith('+00:00'):
        dt_string = dt_string[:-6] + 'Z'
    return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))


def extract_item_dates(
    items_data: List[Dict]
) -> Dict[str, List[Tuple[str, datetime, datetime]]]:

    layer_dates = defaultdict(list)

    for layer in items_data:
        layer_name = layer.get('layer_name')
        if not layer_name:
            continue

        for item_group in layer.get('serialized_item_groups', []):
            for item in item_group:
                item_name = item.get('name')
                geometry = item.get('geometry', {})
                time_range = geometry.get('time_range')

                if not time_range or len(time_range) < 2:
                    print(f"Warning: {item_name} has no valid time_range")
                    continue

                start_time = parse_datetime(time_range[0])
                end_time = parse_datetime(time_range[1])

                layer_dates[layer_name].append(
                    (item_name, start_time, end_time)
                )

    return layer_dates


def validate_patch(patch_dir: Path) -> Dict:
    metadata_path = patch_dir / "metadata.json"
    items_path = patch_dir / "items.json"

    if not metadata_path.exists():
        return {"error": f"metadata.json not found in {patch_dir}"}
    if not items_path.exists():
        return {"error": f"items.json not found in {patch_dir}"}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    with open(items_path, 'r') as f:
        items_data = json.load(f)

    time_range = metadata.get('time_range')
    if not time_range or len(time_range) < 2:
        return {"error": "Invalid time_range in metadata.json"}

    metadata_start = parse_datetime(time_range[0])
    metadata_end = parse_datetime(time_range[1])

    layer_dates = extract_item_dates(items_data)

    results = {
        "patch_name": patch_dir.name,
        "metadata_range": {
            "start": metadata_start.isoformat(),
            "end": metadata_end.isoformat(),
            "days": (metadata_end - metadata_start).days
        },
        "layers": {},
        "issues": []
    }

    for layer_name, items in layer_dates.items():
        layer_results = {
            "total_items": len(items),
            "valid_items": 0,
            "before_range": [],
            "after_range": [],
            "item_dates": []
        }

        for item_name, start_time, end_time in items:
            item_date = start_time
            layer_results["item_dates"].append({
                "name": item_name,
                "date": item_date.isoformat()
            })

            if item_date < metadata_start:
                layer_results["before_range"].append({
                    "name": item_name,
                    "date": item_date.isoformat(),
                    "days_before": (metadata_start - item_date).days
                })
                results["issues"].append(
                    f"{layer_name}: {item_name} is "
                    f"{(metadata_start - item_date).days} days before range start"
                )
            elif item_date > metadata_end:
                layer_results["after_range"].append({
                    "name": item_name,
                    "date": item_date.isoformat(),
                    "days_after": (item_date - metadata_end).days
                })
                results["issues"].append(
                    f"{layer_name}: {item_name} is "
                    f"{(item_date - metadata_end).days} days after range end"
                )
            else:
                layer_results["valid_items"] += 1

        layer_results["item_dates"].sort(key=lambda x: x["date"])

        if layer_results["item_dates"]:
            dates_only = [parse_datetime(d["date"]) for d in layer_results["item_dates"]]
            layer_results["actual_range"] = {
                "start": min(dates_only).isoformat(),
                "end": max(dates_only).isoformat(),
                "days": (max(dates_only) - min(dates_only)).days
            }

            if len(dates_only) > 1:
                gaps = [
                    (dates_only[i + 1] - dates_only[i]).days
                    for i in range(len(dates_only) - 1)
                ]
                layer_results["temporal_gaps"] = {
                    "min_days": min(gaps),
                    "max_days": max(gaps),
                    "avg_days": sum(gaps) / len(gaps)
                }

        results["layers"][layer_name] = layer_results

    return results


def print_results(results: Dict):
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print(f"\nPatch: {results['patch_name']}")

    meta_range = results['metadata_range']
    print("\nMetadata Time Range:")
    print(f"   Start: {meta_range['start']}")
    print(f"   End:   {meta_range['end']}")
    print(f"   Duration: {meta_range['days']} days")

    for layer_name, layer_data in results['layers'].items():
        print(f"\nLayer: {layer_name}")
        print(f"   Total items: {layer_data['total_items']}")
        print(f"   Valid items: {layer_data['valid_items']}")

        if "actual_range" in layer_data:
            actual = layer_data["actual_range"]
            print(
                f"   Actual range: {actual['start']} to "
                f"{actual['end']} ({actual['days']} days)"
            )

        if "temporal_gaps" in layer_data:
            gaps = layer_data["temporal_gaps"]
            print(
                f"   Temporal gaps: min={gaps['min_days']}d, "
                f"max={gaps['max_days']}d, avg={gaps['avg_days']:.1f}d"
            )

        if layer_data["before_range"]:
            print(
                f"   WARNING: {len(layer_data['before_range'])} "
                f"items BEFORE metadata range:"
            )
            for item in layer_data["before_range"][:3]:
                print(f"      - {item['name']}: {item['days_before']} days before")
            if len(layer_data["before_range"]) > 3:
                print(
                    f"      ... and "
                    f"{len(layer_data['before_range']) - 3} more"
                )

        if layer_data["after_range"]:
            print(
                f"   WARNING: {len(layer_data['after_range'])} "
                f"items AFTER metadata range:"
            )
            for item in layer_data["after_range"][:3]:
                print(f"      - {item['name']}: {item['days_after']} days after")
            if len(layer_data["after_range"]) > 3:
                print(
                    f"      ... and "
                    f"{len(layer_data['after_range']) - 3} more"
                )

    if results["issues"]:
        print(f"\nTotal Issues: {len(results['issues'])}")
    else:
        print("\nAll items within metadata time range!")


def validate_all_patches(windows_dir: Path):
    patches = sorted([d for d in windows_dir.iterdir() if d.is_dir()])

    print(f"\nValidating {len(patches)} patches in {windows_dir}")

    all_results = []
    summary = {
        "total_patches": len(patches),
        "patches_with_issues": 0,
        "total_issues": 0
    }

    for patch_dir in patches:
        results = validate_patch(patch_dir)
        all_results.append(results)

        if "error" not in results and results["issues"]:
            summary["patches_with_issues"] += 1
            summary["total_issues"] += len(results["issues"])

        print_results(results)

    print("\nSUMMARY")
    print(f"Total patches: {summary['total_patches']}")
    print(f"Patches with issues: {summary['patches_with_issues']}")
    print(f"Total temporal issues: {summary['total_issues']}")

    if summary["total_issues"] == 0:
        print("All patches have valid temporal consistency!")
    else:
        print(
            f"Found temporal inconsistencies in "
            f"{summary['patches_with_issues']} patches"
        )


def main():
    windows_dir = Path("dataset_pastis/windows/default").expanduser()

    if not windows_dir.exists():
        print(f"ERROR: Directory not found: {windows_dir}")
        return 1

    validate_all_patches(windows_dir)
    return 0


if __name__ == "__main__":
    main()
