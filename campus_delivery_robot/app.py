"""Streamlit app for Campus Delivery Robot Path Planning."""

from __future__ import annotations

from html import escape
from pathlib import Path
import tempfile
import time
from typing import Any

import networkx as nx
import streamlit as st
from streamlit_folium import st_folium

from src.algorithms import ALGORITHMS
from src.graph_builder import apply_route_preferences, build_graph, extract_pois
from src.map_renderer import create_campus_map
from src.osm_parser import parse_osm_file
from src.utils import (
    build_route_table,
    calculate_path_distance,
    format_meters,
    format_seconds,
    get_graph_center,
    summarize_major_places,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OSM_PATH = APP_DIR / "data" / "campus.osm"

ALGORITHM_EXPLANATIONS = {
    "BFS": "BFS finds the route with the fewest graph steps, but it ignores distance and terrain cost.",
    "Uniform Cost Search": "UCS finds the lowest-cost route based on the weighted road-cost model.",
    "A* Search": "A* uses straight-line distance as a heuristic and usually expands fewer nodes than UCS.",
}


def apply_page_style() -> None:
    """Inject presentation-focused Streamlit CSS."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #172033;
        }
        [data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            display: none;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #111827;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] label p,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] .stCaptionContainer p {
            color: #334155 !important;
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: #166534 !important;
        }
        .header-panel {
            background: linear-gradient(135deg, #ffffff 0%, #eef7f4 54%, #fff7ed 100%);
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.1rem;
        }
        .header-panel h1 {
            margin: 0 0 0.3rem 0;
            font-size: 2.05rem;
            line-height: 1.12;
            letter-spacing: 0;
            color: #0f172a;
        }
        .header-panel p {
            margin: 0;
            color: #475569;
            font-size: 1rem;
        }
        .section-panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            margin-top: 1rem;
        }
        .section-panel h3 {
            margin: 0 0 0.45rem 0;
            color: #111827;
            font-size: 1.1rem;
        }
        .section-panel p {
            margin: 0.25rem 0;
            color: #475569;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
        }
        [data-testid="stMetricLabel"] {
            color: #64748b;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 700;
        }
        div.stButton > button:first-child {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #0f766e;
            background: #0f766e;
            color: #ffffff;
            font-weight: 700;
            padding: 0.62rem 1rem;
        }
        div.stButton > button:first-child:hover {
            border-color: #115e59;
            background: #115e59;
            color: #ffffff;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }
        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _bounds_center(metadata: dict[str, Any]) -> tuple[float, float] | None:
    """Return OSM bounds center when bounds are available."""
    bounds = metadata.get("bounds")
    if not bounds:
        return None
    return (
        (float(bounds["minlat"]) + float(bounds["maxlat"])) / 2,
        (float(bounds["minlon"]) + float(bounds["maxlon"])) / 2,
    )


def _build_resources_from_file(osm_path: str | Path, max_pois: int) -> dict[str, Any]:
    """Parse OSM, build the graph, and extract UI-ready POIs."""
    nodes, ways, metadata = parse_osm_file(osm_path)
    graph = build_graph(nodes, ways)
    pois = extract_pois(nodes, ways, graph, max_pois=max_pois)
    center = _bounds_center(metadata) or get_graph_center(graph)

    metadata.update(
        {
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "poi_count": len(pois),
            "skipped_missing_edges": graph.graph.get("skipped_missing_edges", 0),
            "skipped_private_ways": graph.graph.get("skipped_private_ways", 0),
        }
    )

    return {"graph": graph, "pois": pois, "metadata": metadata, "center": center}


@st.cache_resource(show_spinner=False)
def load_resources_from_path(path_str: str, modified_time: float, max_pois: int) -> dict[str, Any]:
    """Cached loader for the default local campus.osm file."""
    resources = _build_resources_from_file(path_str, max_pois=max_pois)
    resources["metadata"]["source_key"] = f"local:{path_str}:{modified_time}"
    resources["metadata"]["source_label"] = "Local data/campus.osm"
    return resources


@st.cache_resource(show_spinner=False)
def load_resources_from_bytes(file_name: str, file_bytes: bytes, max_pois: int) -> dict[str, Any]:
    """Cached loader for uploaded .osm data."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".osm") as tmp:
            tmp.write(file_bytes)
            temp_path = Path(tmp.name)
        resources = _build_resources_from_file(temp_path, max_pois=max_pois)
        resources["metadata"]["source_key"] = f"upload:{file_name}:{len(file_bytes)}"
        resources["metadata"]["source_label"] = f"Uploaded file: {file_name}"
        return resources
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def format_poi_option(poi: dict[str, Any]) -> str:
    """Format a POI for Streamlit select boxes."""
    return str(poi.get("display_name", "Unnamed location"))


def order_pois_for_routing(graph: Any, pois: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort POIs so connected, presentation-friendly choices appear first."""
    component_sizes: dict[str, int] = {}
    for component in nx.connected_components(graph):
        size = len(component)
        for node_id in component:
            component_sizes[str(node_id)] = size

    ordered = []
    for poi in pois:
        item = dict(poi)
        nearest = str(item.get("nearest_graph_node", ""))
        item["component_size"] = component_sizes.get(nearest, 0)
        ordered.append(item)

    return sorted(
        ordered,
        key=lambda item: (
            -int(item.get("component_size", 0)),
            1 if item.get("source") == "fallback_graph_node" else 0,
            str(item.get("display_name", "")).lower(),
        ),
    )


def default_goal_index(pois: list[dict[str, Any]], start_poi: dict[str, Any]) -> int:
    """Choose a default goal from the same connected component when possible."""
    start_node = str(start_poi.get("nearest_graph_node", ""))
    start_component_size = int(start_poi.get("component_size", 0))

    for index, poi in enumerate(pois):
        if str(poi.get("nearest_graph_node", "")) == start_node:
            continue
        if int(poi.get("component_size", 0)) == start_component_size:
            return index

    return 1 if len(pois) > 1 else 0


def run_selected_algorithm(
    base_graph: Any,
    start_node: str,
    goal_node: str,
    algorithm_name: str,
    avoid_service_roads: bool,
    prefer_footways: bool,
) -> dict[str, Any]:
    """Apply route preferences, run a search algorithm, and collect metrics."""
    route_graph = apply_route_preferences(base_graph, avoid_service_roads, prefer_footways)
    search_fn = ALGORITHMS[algorithm_name]

    start_time = time.perf_counter()
    result = search_fn(route_graph, start_node, goal_node)
    running_time = time.perf_counter() - start_time

    path = result.get("path", [])
    result["running_time"] = running_time
    result["algorithm"] = algorithm_name
    result["total_distance"] = calculate_path_distance(route_graph, path) if path else 0.0
    return result


def render_metrics(result: dict[str, Any] | None) -> None:
    """Render the top metric cards."""
    if result and result.get("success"):
        distance = format_meters(float(result.get("total_distance", 0.0)))
        cost = f"{float(result.get('total_cost', 0.0)):.1f}"
        expanded = f"{int(result.get('expanded_nodes', 0)):,}"
        runtime = format_seconds(float(result.get("running_time", 0.0)))
    else:
        distance = "0 m"
        cost = "0"
        expanded = "0"
        runtime = "0 s"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Distance", distance)
    c2.metric("Total Cost", cost)
    c3.metric("Nodes Expanded", expanded)
    c4.metric("Running Time", runtime)


def render_algorithm_explanation(selected_algorithm: str) -> None:
    """Render the algorithm interpretation panel."""
    st.markdown(
        f"""
        <div class="section-panel">
            <h3>Algorithm Explanation</h3>
            <p><strong>{selected_algorithm}</strong>: {ALGORITHM_EXPLANATIONS[selected_algorithm]}</p>
            <p class="small-muted">BFS compares graph depth, UCS compares cumulative weighted cost, and A* adds a straight-line campus-distance heuristic.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Campus Delivery Robot Path Planning",
        page_icon=":material/route:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_page_style()

    st.markdown(
        """
        <div class="header-panel">
            <h1>Campus Delivery Robot Path Planning</h1>
            <p>Search-based route planning using OpenStreetMap data for Central South University.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Route Controls")
        uploaded_file = st.file_uploader("Upload .osm file", type=["osm", "xml"])
        max_pois = st.slider("Maximum POIs", min_value=30, max_value=300, value=150, step=10)
        algorithm_name = st.selectbox("Algorithm", list(ALGORITHMS.keys()), index=2)
        avoid_service_roads = st.checkbox("Avoid service roads", value=False)
        prefer_footways = st.checkbox("Prefer footways", value=True)
        show_graph_nodes = st.checkbox("Show graph nodes", value=False)
        show_route_points = st.checkbox("Show route points", value=False)

    if uploaded_file is None and not DEFAULT_OSM_PATH.exists():
        render_metrics(None)
        st.warning("No local OSM file found. Upload an .osm file from the sidebar to begin.")
        render_algorithm_explanation(algorithm_name)
        return

    try:
        with st.spinner("Building campus graph..."):
            if uploaded_file is not None:
                resources = load_resources_from_bytes(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    max_pois=max_pois,
                )
            else:
                resources = load_resources_from_path(
                    str(DEFAULT_OSM_PATH),
                    DEFAULT_OSM_PATH.stat().st_mtime,
                    max_pois=max_pois,
                )
    except Exception as exc:
        render_metrics(None)
        st.error(f"Could not load the OSM map: {exc}")
        return

    graph = resources["graph"]
    pois = order_pois_for_routing(graph, resources["pois"])
    metadata = resources["metadata"]

    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        render_metrics(None)
        st.error("The OSM file did not produce a usable routing graph. Check that it contains walkable campus roads.")
        return
    if len(pois) < 2:
        render_metrics(None)
        st.error("The map does not contain enough route locations. Try a larger OSM export.")
        return

    with st.sidebar:
        st.success(
            f"Map loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges, {len(pois):,} locations."
        )
        st.caption(metadata.get("source_label", "OSM source ready"))
        start_poi = st.selectbox("Start location", pois, format_func=format_poi_option, index=0)
        goal_default = default_goal_index(pois, start_poi)
        goal_poi = st.selectbox("Goal location", pois, format_func=format_poi_option, index=goal_default)
        generate_clicked = st.button("Generate Route", type="primary")

    start_node = str(start_poi["nearest_graph_node"])
    goal_node = str(goal_poi["nearest_graph_node"])
    selection_key = (
        metadata.get("source_key"),
        start_node,
        goal_node,
        algorithm_name,
        avoid_service_roads,
        prefer_footways,
    )

    if generate_clicked:
        if start_node == goal_node:
            st.session_state["route_result"] = {
                "selection_key": selection_key,
                "success": False,
                "message": "Start and goal resolve to the same graph node. Please choose different locations.",
                "path": [],
                "total_distance": 0.0,
                "total_cost": 0.0,
                "expanded_nodes": 0,
                "running_time": 0.0,
                "algorithm": algorithm_name,
            }
        elif start_node not in graph or goal_node not in graph:
            st.session_state["route_result"] = {
                "selection_key": selection_key,
                "success": False,
                "message": "Start or goal is not connected to the routing graph.",
                "path": [],
                "total_distance": 0.0,
                "total_cost": 0.0,
                "expanded_nodes": 0,
                "running_time": 0.0,
                "algorithm": algorithm_name,
            }
        else:
            with st.spinner("Calculating route..."):
                result = run_selected_algorithm(
                    graph,
                    start_node,
                    goal_node,
                    algorithm_name,
                    avoid_service_roads,
                    prefer_footways,
                )
                result["selection_key"] = selection_key
                if result.get("success"):
                    result["major_places"] = summarize_major_places(pois, result["path"], start_poi, goal_poi)
                else:
                    result["major_places"] = []
                st.session_state["route_result"] = result

    stored_result = st.session_state.get("route_result")
    active_result = stored_result if stored_result and stored_result.get("selection_key") == selection_key else None

    render_metrics(active_result)

    path = active_result.get("path") if active_result and active_result.get("success") else None
    campus_map = create_campus_map(
        graph,
        start_poi=start_poi,
        goal_poi=goal_poi,
        path=path,
        show_graph_nodes=show_graph_nodes,
        show_route_points=show_route_points,
        map_center=resources.get("center"),
    )
    st_folium(campus_map, height=580, width=None, returned_objects=[])

    if active_result:
        if active_result.get("success"):
            st.success(active_result["message"])
            major_places = active_result.get("major_places") or [
                format_poi_option(start_poi),
                format_poi_option(goal_poi),
            ]
            route_path_html = escape(" -> ".join(str(place) for place in major_places))
            st.markdown(
                f"""
                <div class="section-panel">
                    <h3>Route Summary</h3>
                    <p><strong>Route Path:</strong> {route_path_html}</p>
                    <p><strong>Algorithm:</strong> {active_result["algorithm"]}</p>
                    <p><strong>Distance:</strong> {format_meters(float(active_result["total_distance"]))} &nbsp; 
                    <strong>Cost:</strong> {float(active_result["total_cost"]):.1f} &nbsp; 
                    <strong>Nodes Expanded:</strong> {int(active_result["expanded_nodes"]):,} &nbsp; 
                    <strong>Running Time:</strong> {format_seconds(float(active_result["running_time"]))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            route_table = build_route_table(graph, active_result["path"])
            st.subheader("Route Node Sequence")
            st.dataframe(route_table, width="stretch", hide_index=True, height=340)
            with st.expander("Full path node order", expanded=False):
                st.code(" -> ".join(str(node) for node in active_result["path"]))
        else:
            st.warning(active_result.get("message", "No route found."))
    else:
        st.info("Choose a start, goal, and algorithm, then generate a route.")

    render_algorithm_explanation(algorithm_name)

    st.markdown(
        f"""
        <div class="section-panel">
            <h3>Graph Build Details</h3>
            <p>Parsed nodes: {int(metadata.get("node_count", 0)):,} | Parsed ways: {int(metadata.get("way_count", 0)):,} | Routing edges skipped for missing nodes: {int(metadata.get("skipped_missing_edges", 0)):,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
