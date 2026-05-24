"""Folium map rendering helpers."""

from __future__ import annotations

from html import escape
from typing import Any

import folium

from .utils import get_graph_center


ROAD_COLORS = {
    "footway": "#0f766e",
    "pedestrian": "#16a34a",
    "path": "#65a30d",
    "living_street": "#2563eb",
    "service": "#d97706",
    "residential": "#7c3aed",
    "unclassified": "#64748b",
}


def _route_coords(graph: Any, path: list[str] | None) -> list[list[float]]:
    """Convert a node-id path into Folium [lat, lon] coordinates."""
    if not path:
        return []

    coords: list[list[float]] = []
    for node_id in path:
        if node_id not in graph:
            continue
        node_data = graph.nodes[node_id]
        coords.append([float(node_data["lat"]), float(node_data["lon"])])
    return coords


def _add_route_points(campus_map: folium.Map, coords: list[list[float]], max_points: int = 25) -> None:
    """Add small route point markers without overcrowding the map."""
    if len(coords) <= 2:
        return

    step = max(1, len(coords) // max_points)
    for index, coord in enumerate(coords[1:-1:step], start=1):
        folium.CircleMarker(
            location=coord,
            radius=3,
            color="#1d4ed8",
            fill=True,
            fill_color="#93c5fd",
            fill_opacity=0.85,
            opacity=0.85,
            tooltip=f"Route point {index}",
        ).add_to(campus_map)


def _add_graph_overlay(campus_map: folium.Map, graph: Any, max_edges: int = 2500, max_nodes: int = 900) -> None:
    """Draw a lightweight sampled graph overlay for presentation/debugging."""
    edge_items = list(graph.edges(data=True))
    edge_step = max(1, len(edge_items) // max_edges)
    edge_group = folium.FeatureGroup(name="Routing graph", show=True)

    for u, v, data in edge_items[::edge_step]:
        if u not in graph or v not in graph:
            continue
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        highway = data.get("highway", "unclassified")
        folium.PolyLine(
            locations=[
                [float(u_data["lat"]), float(u_data["lon"])],
                [float(v_data["lat"]), float(v_data["lon"])],
            ],
            color=ROAD_COLORS.get(highway, "#64748b"),
            weight=2,
            opacity=0.42,
            tooltip=f"{escape(str(highway))}: {float(data.get('distance', 0.0)):.1f} m",
        ).add_to(edge_group)

    node_items = list(graph.nodes(data=True))
    node_step = max(1, len(node_items) // max_nodes)
    for node_id, data in node_items[::node_step]:
        folium.CircleMarker(
            location=[float(data["lat"]), float(data["lon"])],
            radius=2,
            color="#334155",
            fill=True,
            fill_color="#f8fafc",
            fill_opacity=0.8,
            opacity=0.6,
            tooltip=f"Node {escape(str(node_id))}",
        ).add_to(edge_group)

    edge_group.add_to(campus_map)
    folium.LayerControl(collapsed=True).add_to(campus_map)


def _add_poi_marker(campus_map: folium.Map, poi: dict[str, Any], label: str, color: str, icon: str) -> None:
    """Add a start or goal POI marker."""
    if not poi:
        return

    name = escape(str(poi.get("display_name", label)))
    nearest = escape(str(poi.get("nearest_graph_node", "")))
    popup_html = f"<b>{escape(label)}</b><br>{name}<br>Nearest graph node: {nearest}"
    folium.Marker(
        location=[float(poi["lat"]), float(poi["lon"])],
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"{label}: {name}",
        icon=folium.Icon(color=color, icon=icon),
    ).add_to(campus_map)


def create_campus_map(
    graph: Any,
    start_poi: dict[str, Any] | None = None,
    goal_poi: dict[str, Any] | None = None,
    path: list[str] | None = None,
    show_graph_nodes: bool = False,
    show_route_points: bool = False,
    map_center: tuple[float, float] | None = None,
    zoom_start: int = 16,
) -> folium.Map:
    """
    Create a Folium map with optional start, goal, graph overlay, and route.
    """
    center = map_center or get_graph_center(graph) or (28.172, 112.938)
    campus_map = folium.Map(
        location=[center[0], center[1]],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if show_graph_nodes:
        _add_graph_overlay(campus_map, graph)

    route_coords = _route_coords(graph, path)
    if len(route_coords) >= 2:
        folium.PolyLine(route_coords, color="#0f172a", weight=10, opacity=0.45).add_to(campus_map)
        folium.PolyLine(
            route_coords,
            color="#ef4444",
            weight=6,
            opacity=0.96,
            tooltip="Calculated robot route",
        ).add_to(campus_map)
        if show_route_points:
            _add_route_points(campus_map, route_coords)

    if start_poi:
        _add_poi_marker(campus_map, start_poi, "Start", "green", "play")
    if goal_poi:
        _add_poi_marker(campus_map, goal_poi, "Goal", "red", "flag")

    bounds: list[list[float]] = []
    bounds.extend(route_coords)
    for poi in (start_poi, goal_poi):
        if poi:
            bounds.append([float(poi["lat"]), float(poi["lon"])])
    if len(bounds) >= 2:
        campus_map.fit_bounds(bounds, padding=(28, 28))

    return campus_map

