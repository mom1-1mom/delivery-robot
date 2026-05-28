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

SEGMENT_COLORS = [
    "#16a34a",
    "#2563eb",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#dc2626",
]


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


def _edge_coords(graph: Any, edge: tuple[str, str]) -> list[list[float]] | None:
    """Convert an explored edge into Folium coordinates."""
    u, v = str(edge[0]), str(edge[1])
    if u not in graph or v not in graph:
        return None
    u_data = graph.nodes[u]
    v_data = graph.nodes[v]
    return [
        [float(u_data["lat"]), float(u_data["lon"])],
        [float(v_data["lat"]), float(v_data["lon"])],
    ]


def _sample_edges(edges: list[tuple[str, str]], max_edges: int) -> list[tuple[str, str]]:
    """Sample explored edges while preserving broad search progress order."""
    if max_edges <= 0:
        return []
    if len(edges) <= max_edges:
        return edges
    step = max(1, len(edges) // max_edges)
    return edges[::step][:max_edges]


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


def _numbered_stop_marker(campus_map: folium.Map, poi: dict[str, Any], stop_number: int) -> None:
    """Add a numbered delivery stop marker."""
    name = escape(str(poi.get("display_name", f"Stop {stop_number}")))
    nearest = escape(str(poi.get("nearest_graph_node", "")))
    icon_html = f"""
    <div style="
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #2563eb;
        color: #ffffff;
        border: 2px solid #ffffff;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 800;">
        {stop_number}
    </div>
    """
    folium.Marker(
        location=[float(poi["lat"]), float(poi["lon"])],
        popup=folium.Popup(f"<b>Stop {stop_number}</b><br>{name}<br>Nearest graph node: {nearest}", max_width=280),
        tooltip=f"Stop {stop_number}: {name}",
        icon=folium.DivIcon(html=icon_html, icon_size=(30, 30), icon_anchor=(15, 15)),
    ).add_to(campus_map)


def _add_direction_marker(campus_map: folium.Map, coords: list[list[float]], label: str) -> None:
    """Add a compact direction marker near the middle of a route segment."""
    if len(coords) < 2:
        return
    midpoint = coords[len(coords) // 2]
    folium.CircleMarker(
        location=midpoint,
        radius=5,
        color="#ffffff",
        fill=True,
        fill_color="#0f766e",
        fill_opacity=0.95,
        opacity=0.9,
        tooltip=label,
    ).add_to(campus_map)


def render_multi_stop_map(
    nodes: Any,
    route_result: dict[str, Any] | None,
    poi_lookup: dict[str, dict[str, Any]] | None,
    show_search_process: bool = False,
    search_progress: int = 100,
    final_route_only: bool = True,
    max_explored_edges: int = 1000,
) -> folium.Map:
    """
    Render a multi-stop delivery map.

    The first parameter is the routing graph. It is named `nodes` to match the
    project brief, but the object is expected to be a NetworkX graph with node
    coordinate attributes.
    """
    graph = nodes
    del poi_lookup

    full_path = route_result.get("full_path", []) if route_result else []
    start_poi = route_result.get("start_poi") if route_result else None
    delivery_order = route_result.get("delivery_order", []) if route_result else []
    segments = route_result.get("segments", []) if route_result else []

    route_coords = _route_coords(graph, full_path)
    center = None
    if route_coords:
        center = (
            sum(coord[0] for coord in route_coords) / len(route_coords),
            sum(coord[1] for coord in route_coords) / len(route_coords),
        )
    else:
        center = get_graph_center(graph) or (28.172, 112.938)

    campus_map = folium.Map(
        location=[center[0], center[1]],
        zoom_start=16,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if route_result and show_search_process and not final_route_only:
        explored_edges = route_result.get("all_explored_edges", [])
        sampled_edges = _sample_edges(explored_edges, max_explored_edges)
        progress = min(100, max(0, int(search_progress)))
        visible_count = int(len(sampled_edges) * progress / 100)
        visible_edges = sampled_edges[:visible_count]
        current_batch_start = max(0, visible_count - max(20, visible_count // 8))

        explored_group = folium.FeatureGroup(name="Search process", show=True)
        for index, edge in enumerate(visible_edges):
            coords = _edge_coords(graph, edge)
            if not coords:
                continue
            is_current = index >= current_batch_start
            folium.PolyLine(
                locations=coords,
                color="#f59e0b" if is_current else "#93c5fd",
                weight=3 if is_current else 2,
                opacity=0.55 if is_current else 0.28,
            ).add_to(explored_group)
        explored_group.add_to(campus_map)

    for segment in segments:
        coords = _route_coords(graph, segment.get("path", []))
        if len(coords) < 2:
            continue
        color = SEGMENT_COLORS[(int(segment.get("segment", 1)) - 1) % len(SEGMENT_COLORS)]
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=5,
            opacity=0.52,
            tooltip=f"Segment {segment.get('segment')}: {escape(str(segment.get('from_name', '')))} to {escape(str(segment.get('to_name', '')))}",
        ).add_to(campus_map)
        _add_direction_marker(campus_map, coords, f"Direction: segment {segment.get('segment')}")

    if len(route_coords) >= 2:
        folium.PolyLine(route_coords, color="#052e16", weight=10, opacity=0.32).add_to(campus_map)
        folium.PolyLine(
            route_coords,
            color="#16a34a",
            weight=6,
            opacity=0.96,
            tooltip="Final multi-stop delivery route",
        ).add_to(campus_map)

    if start_poi:
        _add_poi_marker(campus_map, start_poi, "Start", "green", "play")

    for index, poi in enumerate(delivery_order, start=1):
        _numbered_stop_marker(campus_map, poi, index)

    bounds: list[list[float]] = []
    bounds.extend(route_coords)
    for poi in [start_poi, *delivery_order]:
        if poi:
            bounds.append([float(poi["lat"]), float(poi["lon"])])
    if len(bounds) >= 2:
        campus_map.fit_bounds(bounds, padding=(30, 30))

    return campus_map
