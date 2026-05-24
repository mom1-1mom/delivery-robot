"""Shared utility functions for map geometry and route reporting."""

from __future__ import annotations

import math
from typing import Any, Iterable


EARTH_RADIUS_M = 6_371_000


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two lat/lon points in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def calculate_path_distance(graph: Any, path: Iterable[str]) -> float:
    """Sum edge distances for a route represented by graph node ids."""
    node_list = list(path)
    distance = 0.0
    for u, v in zip(node_list, node_list[1:]):
        if graph.has_edge(u, v):
            distance += float(graph[u][v].get("distance", 0.0))
    return distance


def calculate_path_cost(graph: Any, path: Iterable[str]) -> float:
    """Sum edge costs for a route represented by graph node ids."""
    node_list = list(path)
    cost = 0.0
    for u, v in zip(node_list, node_list[1:]):
        if graph.has_edge(u, v):
            cost += float(graph[u][v].get("cost", graph[u][v].get("distance", 0.0)))
    return cost


def get_graph_center(graph: Any) -> tuple[float, float] | None:
    """Return the average lat/lon of graph nodes, or None for an empty graph."""
    if graph.number_of_nodes() == 0:
        return None

    lat_sum = 0.0
    lon_sum = 0.0
    count = 0
    for _, data in graph.nodes(data=True):
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is None or lon is None:
            continue
        lat_sum += float(lat)
        lon_sum += float(lon)
        count += 1

    if count == 0:
        return None
    return lat_sum / count, lon_sum / count


def build_route_table(graph: Any, path: list[str]):
    """Build a Pandas table with route node coordinates."""
    import pandas as pd

    rows = []
    for step, node_id in enumerate(path, start=1):
        data = graph.nodes[node_id]
        rows.append(
            {
                "Step": step,
                "Node ID": node_id,
                "Latitude": round(float(data.get("lat", 0.0)), 7),
                "Longitude": round(float(data.get("lon", 0.0)), 7),
            }
        )
    return pd.DataFrame(rows)


def format_meters(value: float) -> str:
    """Format meters for compact presentation."""
    if value >= 1000:
        return f"{value / 1000:.2f} km"
    return f"{value:.1f} m"


def format_seconds(value: float) -> str:
    """Format a runtime value in seconds."""
    if value < 0.001:
        return "<0.001 s"
    return f"{value:.4f} s"


def summarize_major_places(
    pois: list[dict[str, Any]],
    path: list[str],
    start_poi: dict[str, Any],
    goal_poi: dict[str, Any],
    max_items: int = 8,
) -> list[str]:
    """Return named POIs whose nearest graph nodes are on the route."""
    if not path:
        return []

    path_set = set(path)
    route_node_to_names: dict[str, list[str]] = {}
    for poi in pois:
        nearest = poi.get("nearest_graph_node")
        name = poi.get("display_name")
        if nearest in path_set and name:
            route_node_to_names.setdefault(nearest, []).append(str(name))

    ordered_names: list[str] = []
    seen = set()

    def add_name(name: str) -> None:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)

    add_name(str(start_poi.get("display_name", "Start")))
    for node_id in path:
        for name in route_node_to_names.get(node_id, [])[:2]:
            add_name(name)
            if len(ordered_names) >= max_items - 1:
                break
        if len(ordered_names) >= max_items - 1:
            break
    add_name(str(goal_poi.get("display_name", "Goal")))

    return ordered_names[:max_items]

