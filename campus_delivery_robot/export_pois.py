"""Export OSM-derived POIs to a human-editable whitelist CSV.

Run:
    python export_pois.py --overwrite

Then edit:
    data/poi_whitelist.csv

The Streamlit app reads enabled rows from this CSV before falling back to raw
OSM POI extraction.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import networkx as nx

from src.graph_builder import build_graph, extract_pois
from src.osm_parser import parse_osm_file


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OSM_PATH = APP_DIR / "data" / "campus.osm"
DEFAULT_OUTPUT_PATH = APP_DIR / "data" / "poi_whitelist.csv"

ENABLED_ENGLISH_NAMES = [
    "Building A Bicycle Parking Lot",
    "Central South University Lunan Campus",
    "Building 24",
    "Building A",
    "Building B",
    "Building C",
    "Building D",
    "CSU Security Office",
    "CSU Logistics Maintenance Service Center",
    "CSU Swimming, Badminton and Basketball Gym",
    "CSU Swimming, Badminton and Basketball Gym 2",
    "CSU School of Socialism",
    "Yunluyuan Community Service Center",
    "School of Chemistry and Chemical Engineering",
    "Building B Canteen",
    "No. 7 Canteen Xiyangyang Supermarket",
    "CSU No. 7 Canteen",
    "CSU No. 2 Canteen",
    "CSU No. 8 Canteen",
    "No. 2 Canteen Parking Lot",
    "CSU Student Dormitory",
    "Shenghua Dormitory Buildings 37 and 38",
    "Shenghua Dormitory Clinic",
    "Dormitory Building 1 North",
    "Dormitory Building 10",
    "Dormitory Building 11",
    "Dormitory Building 13",
    "Dormitory Building 14",
    "Dormitory Building 15",
    "Dormitory Building 16",
    "Dormitory Building 18",
    "Dormitory Building 2",
    "Dormitory Building 20",
    "Dormitory Building 21",
    "Dormitory Building 24",
    "Dormitory Building 25",
    "Dormitory Building 26",
    "Dormitory Building 28 North",
    "Dormitory Building 28 South",
    "Dormitory Building 29",
    "Dormitory Building 31",
    "Dormitory Building 5",
    "Dormitory Building 6",
    "Dormitory Building 8",
    "Dormitory Building 9",
    "Dormitory Building 17",
    "Shenghua West Entrance",
    "CSU South Campus North Gate",
    "CSU New Campus North Gate",
    "South Gate",
    "New Campus Stadium West Gate",
    "CSU Library Front Square",
    "CSU New Campus Library",
    "Library Front Bicycle Parking Area",
    "Huanghe Hospital",
    "CSU South Campus Security Office",
    "China Post",
    "Chunzhiniao Supermarket",
    "New Campus Stadium",
    "CSU Comprehensive Building",
    "CSU Lecture Hall",
    "Bingqing Building",
    "Mechanical and Electrical Building Bicycle Parking Area",
    "Zuojialong West",
    "Yanyutang Pond",
    "Mozishan Hill",
]


NOISY_NAME_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^\d+\s*\u53f7?$"),
    re.compile(r".*\u8def\u53e3$"),
    re.compile(r".*\u516c\u4ea4.*"),
    re.compile(r".*\u5730\u94c1.*"),
]

CATEGORY_KEYWORDS = [
    ("library", ["library", "\u56fe\u4e66\u9986"]),
    ("canteen", ["canteen", "\u98df\u5802"]),
    ("gate", ["gate", "entrance", "\u95e8"]),
    ("dormitory", ["dormitory", "apartment", "\u516c\u5bd3", "\u5bbf\u820d"]),
    ("teaching_building", ["teaching", "lecture", "hall", "\u6559\u5b66", "\u8bb2\u5802", "\u697c"]),
    ("office", ["office", "\u529e\u516c", "\u4fdd\u536b\u5904"]),
    ("service", ["post", "bank", "mobile", "\u90ae\u653f", "\u94f6\u884c", "\u8425\u4e1a\u5385"]),
    ("shop", ["shop", "supermarket", "store", "\u8d85\u5e02", "\u4fbf\u5229\u5e97"]),
    ("food", ["restaurant", "kfc", "burger", "noodles", "\u9910", "\u996d", "\u7c89", "\u9762"]),
    ("sports", ["sports", "stadium", "gym", "\u4f53\u80b2", "\u8fd0\u52a8", "\u64cd\u573a"]),
    ("medical", ["hospital", "clinic", "\u533b\u9662", "\u8bca\u6240"]),
]


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="Export campus POIs to a whitelist CSV.")
    parser.add_argument("--osm", default=str(DEFAULT_OSM_PATH), help="Path to the input .osm file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to the output CSV file.")
    parser.add_argument("--max-pois", type=int, default=300, help="Maximum POIs to export.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing CSV file.")
    return parser.parse_args()


def infer_category(name: str, source: str) -> str:
    """Infer a practical category from a POI name and source type."""
    lowered = name.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword.lower() in lowered for keyword in keywords):
            return category

    if "highway" in source:
        return "road_or_stop"
    if "building" in source:
        return "building"
    if "amenity" in source:
        return "amenity"
    if "shop" in source:
        return "shop"
    return "other"


def is_noisy_name(name: str) -> bool:
    """Return True when a POI name is likely too noisy for delivery demos."""
    stripped = name.strip()
    if len(stripped) <= 1:
        return True
    return any(pattern.match(stripped) for pattern in NOISY_NAME_PATTERNS)


def component_sizes(graph: nx.Graph) -> dict[str, int]:
    """Map each graph node id to its connected-component size."""
    sizes: dict[str, int] = {}
    # Component size is used to exclude POIs outside the main routing network.
    for component in nx.connected_components(graph):
        size = len(component)
        for node_id in component:
            sizes[str(node_id)] = size
    return sizes


def should_enable_by_default(poi: dict[str, Any], largest_component_size: int) -> tuple[int, str]:
    """Suggest whether a POI should initially be enabled."""
    name = str(poi.get("display_name", ""))
    source = str(poi.get("source", ""))
    category = infer_category(name, source)
    component_size = int(poi.get("component_size", 0))

    if component_size < largest_component_size:
        return 0, "outside_largest_routing_component"
    if is_noisy_name(name):
        return 0, "noisy_or_numeric_name"
    if category in {"road_or_stop", "other"}:
        return 0, "low_value_for_delivery_demo"
    return 1, "recommended"


def apply_default_english_names(rows: list[dict[str, Any]]) -> None:
    """Apply English names to the default enabled rows produced by this map."""
    enabled_rows = [row for row in rows if int(row["enabled"]) == 1]
    if len(enabled_rows) != len(ENABLED_ENGLISH_NAMES):
        return
    for row, english_name in zip(enabled_rows, ENABLED_ENGLISH_NAMES):
        row["display_name"] = english_name


def build_rows(osm_path: Path, max_pois: int) -> list[dict[str, Any]]:
    """Parse OSM data and build CSV-ready POI rows."""
    nodes, ways, _ = parse_osm_file(osm_path)
    graph = build_graph(nodes, ways)
    pois = extract_pois(nodes, ways, graph, max_pois=max_pois, min_named_pois=10, fallback_count=60)
    sizes = component_sizes(graph)
    largest_component_size = max(sizes.values(), default=0)

    rows: list[dict[str, Any]] = []
    # Preserve both editable display names and original OSM names in the CSV.
    for poi in pois:
        nearest_node = str(poi.get("nearest_graph_node", ""))
        component_size = sizes.get(nearest_node, 0)
        poi["component_size"] = component_size

        original_name = str(poi.get("display_name", ""))
        category = infer_category(original_name, str(poi.get("source", "")))
        enabled, note = should_enable_by_default(poi, largest_component_size)

        rows.append(
            {
                "enabled": enabled,
                "display_name": original_name,
                "original_name": original_name,
                "category": category,
                "osm_id": poi.get("osm_id", ""),
                "source": poi.get("source", ""),
                "lat": f"{float(poi.get('lat', 0.0)):.7f}",
                "lon": f"{float(poi.get('lon', 0.0)):.7f}",
                "nearest_graph_node": nearest_node,
                "nearest_distance_m": f"{float(poi.get('nearest_distance_m', 0.0)):.1f}",
                "component_size": component_size,
                "note": note,
            }
        )

    # Place recommended, well-connected POIs first for easier manual review.
    rows.sort(key=lambda row: (-int(row["enabled"]), -int(row["component_size"]), row["category"], row["display_name"]))
    apply_default_english_names(rows)
    return rows


def write_csv(output_path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    """Write rows to a CSV file."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite if you want to regenerate it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "enabled",
        "display_name",
        "original_name",
        "category",
        "osm_id",
        "source",
        "lat",
        "lon",
        "nearest_graph_node",
        "nearest_distance_m",
        "component_size",
        "note",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Export POIs to CSV."""
    args = parse_args()
    osm_path = Path(args.osm)
    output_path = Path(args.output)

    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    rows = build_rows(osm_path, args.max_pois)
    write_csv(output_path, rows, args.overwrite)

    enabled_count = sum(1 for row in rows if int(row["enabled"]) == 1)
    print(f"Exported {len(rows)} POIs to {output_path}")
    print(f"Suggested enabled POIs: {enabled_count}")
    print("Edit enabled/display_name/category in the CSV, then save it as UTF-8 CSV.")


if __name__ == "__main__":
    main()
