"""Streamlit app for multi-stop campus delivery robot route planning."""

from __future__ import annotations

import csv
from html import escape
from io import BytesIO
from pathlib import Path
import tempfile
from typing import Any

import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.congestion_model import TrafficCongestionModel

from src.algorithms import ALGORITHMS
from src.graph_builder import apply_route_preferences, build_graph, extract_pois
from src.map_renderer import render_multi_stop_map
from src.multistop_planner import plan_multi_stop_route
from src.osm_parser import parse_osm_file
from src.utils import build_route_table, format_meters, format_seconds, get_graph_center


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OSM_PATH = APP_DIR / "data" / "campus.osm"
DEFAULT_POI_WHITELIST_PATH = APP_DIR / "data" / "poi_whitelist.csv"
MAX_DELIVERY_POINTS = 8

ALGORITHM_EXPLANATIONS = {
    "BFS": "BFS finds a route with the fewest graph steps, but it does not consider weighted delivery cost.",
    "Uniform Cost Search": "Uniform Cost Search finds the lowest-cost route based on edge weights such as road type and distance.",
    "A* Search": "A* uses a heuristic based on straight-line distance to guide the search, often reducing the number of expanded nodes.",
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
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            display: none;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] label p,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] .stCaptionContainer p {
            color: #000000 !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] > div > div {
            border: 1px solid #000000 !important;
            background: #000000 !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] span,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] div {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: #166534 !important;
        }
        .hero-panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
            margin-bottom: 1.1rem;
        }
        .hero-panel h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.15rem;
            line-height: 1.12;
            letter-spacing: 0;
            color: #0f172a;
        }
        .hero-panel p {
            margin: 0.25rem 0;
            color: #475569;
            font-size: 1rem;
        }
        .section-panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
            margin-top: 1rem;
        }
        .section-panel h3 {
            margin: 0 0 0.5rem 0;
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
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.055);
        }
        [data-testid="stMetricLabel"] {
            color: #64748b;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
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
        .inline-heading {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0 0 0.4rem 0;
            color: #111827;
            font-weight: 800;
            font-size: 1.05rem;
        }
        .inline-heading span {
            color: #0f766e;
            font-size: 1rem;
            line-height: 1;
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


def _enabled(value: str) -> bool:
    """Return True for common enabled values in a CSV whitelist."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_pois_from_whitelist(whitelist_path: str | Path, graph: Any) -> list[dict[str, Any]]:
    """Load hand-cleaned POIs from data/poi_whitelist.csv."""
    path = Path(whitelist_path)
    if not path.exists():
        return []

    pois: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if not _enabled(row.get("enabled", "")):
                continue

            nearest_node = str(row.get("nearest_graph_node", "")).strip()
            if not nearest_node or nearest_node not in graph or nearest_node in seen_nodes:
                continue

            node_data = graph.nodes[nearest_node]
            try:
                lat = float(row.get("lat") or node_data["lat"])
                lon = float(row.get("lon") or node_data["lon"])
            except (TypeError, ValueError):
                lat = float(node_data["lat"])
                lon = float(node_data["lon"])

            pois.append(
                {
                    "display_name": str(row.get("display_name") or row.get("original_name") or nearest_node),
                    "original_name": str(row.get("original_name") or row.get("display_name") or ""),
                    "category": str(row.get("category") or "manual"),
                    "osm_id": str(row.get("osm_id") or nearest_node),
                    "source": str(row.get("source") or "poi_whitelist"),
                    "lat": lat,
                    "lon": lon,
                    "nearest_graph_node": nearest_node,
                    "nearest_distance_m": float(row.get("nearest_distance_m") or 0.0),
                }
            )
            seen_nodes.add(nearest_node)

    return pois


def _build_resources_from_file(
    osm_path: str | Path,
    max_pois: int,
    whitelist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Parse OSM, build the graph, and extract UI-ready POIs."""
    nodes, ways, metadata = parse_osm_file(osm_path)
    graph = build_graph(nodes, ways)
    whitelist_pois = _load_pois_from_whitelist(whitelist_path, graph) if whitelist_path else []
    if whitelist_pois:
        pois = whitelist_pois[:max_pois]
        poi_source = "Cleaned POI whitelist"
    else:
        pois = extract_pois(nodes, ways, graph, max_pois=max_pois)
        poi_source = "Automatic OSM extraction"
    center = _bounds_center(metadata) or get_graph_center(graph)
    positions = [
        (str(node_id), float(data["lat"]), float(data["lon"]))
        for node_id, data in graph.nodes(data=True)
        if data.get("lat") is not None and data.get("lon") is not None
    ]

    metadata.update(
        {
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "poi_count": len(pois),
            "poi_source": poi_source,
            "skipped_missing_edges": graph.graph.get("skipped_missing_edges", 0),
            "skipped_private_ways": graph.graph.get("skipped_private_ways", 0),
        }
    )

    return {"nodes": nodes, "graph": graph, "pois": pois, "positions": positions, "metadata": metadata, "center": center}


@st.cache_resource(show_spinner=False)
def load_resources_from_path(
    path_str: str,
    modified_time: float,
    max_pois: int,
    whitelist_path_str: str,
    whitelist_modified_time: float,
) -> dict[str, Any]:
    """Cached loader for the default local campus.osm file."""
    resources = _build_resources_from_file(path_str, max_pois=max_pois, whitelist_path=whitelist_path_str)
    resources["metadata"]["source_key"] = f"local:{path_str}:{modified_time}:whitelist:{whitelist_modified_time}"
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


@st.cache_resource(show_spinner=False)
def load_traffic_model_from_csv(file_name: str, file_bytes: bytes) -> TrafficCongestionModel:
    """Train and cache a traffic model from uploaded historical congestion data."""
    dataframe = pd.read_csv(BytesIO(file_bytes))
    model = TrafficCongestionModel()
    model.train_from_dataframe(dataframe)
    return model


def format_train_rmse(value: float | None) -> str:
    """Format the model validation RMSE for display."""
    return f"{value:.1f} sec" if value is not None else "N/A"


def render_congestion_model_summary(model: TrafficCongestionModel | None, hour: int, weekday: int = 0) -> None:
    """Render the traffic model training summary in the sidebar."""
    if model is None or not model.trained:
        return

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    st.markdown("### Traffic Model Status")
    st.write(
        f"- Trained on **{model.train_samples:,} samples**  \n"
        f"- Validation RMSE: **{format_train_rmse(model.train_rmse)}**  \n"
        f"- Route costs use predicted travel time for **{weekday_names[weekday]} at {hour}:00**."
    )


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


def format_minutes(value: float) -> str:
    """Format minutes for display."""
    if value <= 0:
        return "0 min"
    if value < 1:
        return "<1 min"
    return f"{value:.1f} min"


def format_fee(value: float) -> str:
    """Format delivery fee in CNY."""
    return f"¥{value:.2f}"


def ensure_delivery_slot_state() -> None:
    """Initialize dynamic delivery point controls."""
    if "delivery_slot_count" not in st.session_state:
        st.session_state["delivery_slot_count"] = 1


def add_delivery_slot() -> None:
    """Add one delivery input slot up to the project maximum."""
    st.session_state["delivery_slot_count"] = min(
        MAX_DELIVERY_POINTS,
        int(st.session_state.get("delivery_slot_count", 1)) + 1,
    )


def clear_delivery_slots() -> None:
    """Reset delivery point controls."""
    st.session_state["delivery_slot_count"] = 1
    for key in list(st.session_state.keys()):
        if str(key).startswith("delivery_poi_"):
            del st.session_state[key]


def selected_delivery_pois(start_poi: dict[str, Any], pois: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render dynamic delivery point select boxes and return selected POIs."""
    start_node = str(start_poi.get("nearest_graph_node", ""))
    options = [poi for poi in pois if str(poi.get("nearest_graph_node", "")) != start_node]
    selected: list[dict[str, Any]] = []

    if not options:
        st.warning("No delivery points are available for the selected start location.")
        return selected

    st.markdown("Delivery points")
    for index in range(int(st.session_state["delivery_slot_count"])):
        key = f"delivery_poi_{index}"
        if key in st.session_state and st.session_state[key] not in options:
            del st.session_state[key]
        default_index = min(index, len(options) - 1)
        selected.append(
            st.selectbox(
                f"Delivery point {index + 1}",
                options,
                format_func=format_poi_option,
                index=default_index,
                key=key,
            )
        )

    add_col, clear_col = st.columns(2)
    add_col.button("+ Add Stop", on_click=add_delivery_slot, disabled=int(st.session_state["delivery_slot_count"]) >= MAX_DELIVERY_POINTS)
    clear_col.button("Reset Stops", on_click=clear_delivery_slots)
    return selected


def validate_delivery_selection(start_poi: dict[str, Any], delivery_pois: list[dict[str, Any]]) -> str | None:
    """Validate selected delivery POIs before running the planner."""
    if not delivery_pois:
        return "Select at least one delivery point."
    if len(delivery_pois) > MAX_DELIVERY_POINTS:
        return f"Select at most {MAX_DELIVERY_POINTS} delivery points."

    start_node = str(start_poi.get("nearest_graph_node", ""))
    seen_nodes: set[str] = set()
    for index, poi in enumerate(delivery_pois, start=1):
        node = poi.get("nearest_graph_node")
        if node is None:
            return f"Delivery point {index} does not have a nearest graph node."
        node = str(node)
        if node == start_node:
            return "Start location cannot also be a delivery point."
        if node in seen_nodes:
            return "Two selected delivery points resolve to the same graph node."
        seen_nodes.add(node)

    return None


def render_business_metrics(result: dict[str, Any] | None) -> None:
    """Render user-facing delivery metrics first."""
    if result and result.get("success"):
        estimated_time = format_minutes(float(result.get("estimated_time", 0.0)))
        distance = format_meters(float(result.get("total_distance", 0.0)))
        fee = format_fee(float(result.get("delivery_fee", 0.0)))
    else:
        estimated_time = "0 min"
        distance = "0 m"
        fee = "¥0.00"

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Delivery Time", estimated_time)
    c2.metric("Total Distance", distance)
    c3.metric("Predicted Fee", fee)


def render_technical_metrics(result: dict[str, Any] | None, algorithm_name: str) -> None:
    """Render technical algorithm metrics after the business summary."""
    if result and result.get("success"):
        total_cost = f"{float(result.get('total_cost', 0.0)):.1f}"
        runtime = format_seconds(float(result.get("running_time", 0.0)))
        expanded = f"{int(result.get('total_expanded_nodes', 0)):,}"
        route_nodes = f"{len(result.get('full_path', [])):,}"
        deliveries = str(len(result.get("delivery_order", [])))
    else:
        total_cost = "0"
        runtime = "0 s"
        expanded = "0"
        route_nodes = "0"
        deliveries = "0"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost", total_cost)
    c2.metric("Running Time", runtime)
    c3.metric("Nodes Expanded", expanded)
    c4.metric("Algorithm", algorithm_name)

    c5, c6 = st.columns(2)
    c5.metric("Graph Nodes in Route", route_nodes)
    c6.metric("Delivery Points", deliveries)


def build_preview_result(start_poi: dict[str, Any], delivery_pois: list[dict[str, Any]], algorithm_name: str) -> dict[str, Any]:
    """Build a marker-only route result before planning."""
    return {
        "success": False,
        "start_poi": start_poi,
        "delivery_order": delivery_pois,
        "delivery_pois": delivery_pois,
        "full_path": [],
        "segments": [],
        "all_explored_edges": [],
        "algorithm": algorithm_name,
    }


def route_selection_key(
    metadata: dict[str, Any],
    start_poi: dict[str, Any],
    delivery_pois: list[dict[str, Any]],
    algorithm_name: str,
    prefer_footways: bool,
    departure_time: tuple[int, int] | None = None,
) -> tuple[Any, ...]:
    """Build a cache key for the currently selected route problem."""
    delivery_nodes = tuple(str(poi.get("nearest_graph_node", "")) for poi in delivery_pois)
    return (
        metadata.get("source_key"),
        str(start_poi.get("nearest_graph_node", "")),
        delivery_nodes,
        algorithm_name,
        prefer_footways,
        departure_time,
    )


def render_delivery_order(result: dict[str, Any]) -> None:
    """Render the planned delivery order."""
    if not result.get("success"):
        return
    names = ["Start"]
    names.extend(f"Stop {index}: {format_poi_option(poi)}" for index, poi in enumerate(result["delivery_order"], start=1))
    order_html = escape(" -> ".join(names))
    st.markdown(
        f"""
        <div class="section-panel">
            <div class="inline-heading"><span>↳</span> Delivery Order</div>
            <p>{order_html}</p>
            <p class="small-muted">Planning method: {escape(str(result.get("planning_method", "Route optimisation")))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_segment_table(result: dict[str, Any]) -> pd.DataFrame:
    """Build a display table for planned route segments."""
    rows = []
    for segment in result.get("segments", []):
        rows.append(
            {
                "Segment": int(segment.get("segment", 0)),
                "From": segment.get("from_name", ""),
                "To": segment.get("to_name", ""),
                "Distance": format_meters(float(segment.get("distance", 0.0))),
                "Cost": f"{float(segment.get('cost', 0.0)):.1f}",
                "Expanded Nodes": int(segment.get("expanded_nodes", 0)),
            }
        )
    return pd.DataFrame(rows)


def render_algorithm_explanation(selected_algorithm: str) -> None:
    """Render the algorithm interpretation panel."""
    st.markdown(
        f"""
        <div class="section-panel">
            <div class="inline-heading"><span>i</span> Algorithm Interpretation</div>
            <p><strong>{escape(selected_algorithm)}</strong>: {escape(ALGORITHM_EXPLANATIONS[selected_algorithm])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Campus Delivery Robot Route Planner",
        page_icon=":material/route:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_page_style()
    ensure_delivery_slot_state()

    st.markdown(
        """
        <div class="hero-panel">
            <h1>Campus Delivery Robot Route Planner</h1>
            <p>Multi-stop search-based route planning using OpenStreetMap campus data.</p>
            <p>This system plans delivery routes for a campus robot using BFS, Uniform Cost Search and A* Search.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Route Controls")
        uploaded_file = None
        st.caption("Using the default campus map data.")
        max_pois = st.slider("Maximum POIs", min_value=40, max_value=300, value=180, step=10)
        algorithm_name = st.selectbox("Algorithm", list(ALGORITHMS.keys()), index=2)

        st.markdown("Route model")
        prefer_footways = st.checkbox("Prefer footways", value=True)

        congestion_model: TrafficCongestionModel | None = None
        departure_hour = 8
        departure_weekday = 0
        with st.expander("Traffic congestion model", expanded=False):
            congestion_csv = st.file_uploader("Upload historical congestion CSV", type=["csv"], key="congestion_csv")
            if congestion_csv is not None:
                with st.spinner("Training traffic model..."):
                    try:
                        congestion_model = load_traffic_model_from_csv(congestion_csv.name, congestion_csv.getvalue())
                        st.success("Traffic model trained.")
                    except Exception as exc:
                        st.error(f"Could not train traffic model: {exc}")
            show_congestion_overlay = False
            if congestion_model is not None and congestion_model.trained:
                col1, col2 = st.columns(2)
                with col1:
                    departure_hour = st.slider("Departure hour", min_value=0, max_value=23, value=8, step=1)
                with col2:
                    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    departure_weekday = st.selectbox("Day of week", list(range(7)), format_func=lambda x: weekday_names[x], index=0)

                show_congestion_overlay = st.checkbox(
                    "Show predicted congestion overlay",
                    value=True,
                    help="Display edge colors from green to red based on predicted congestion at the selected time.",
                )

            render_congestion_model_summary(congestion_model, departure_hour, departure_weekday)

        st.markdown("Search Process Visualization")
        show_search_process = st.checkbox("Show search process", value=True)
        final_route_only = st.checkbox("Final route only", value=False)
        search_progress = 100
        if show_search_process and not final_route_only:
            search_progress = st.slider("Search progress", min_value=0, max_value=100, value=100, step=5)
        max_explored_edges = st.slider("Max explored edges", min_value=100, max_value=3000, value=1000, step=100)

    if uploaded_file is None and not DEFAULT_OSM_PATH.exists():
        render_business_metrics(None)
        st.warning("No local OSM file found. Upload an .osm file from the sidebar to begin.")
        render_algorithm_explanation(algorithm_name)
        return

    try:
        with st.spinner("Building campus graph..."):
            if uploaded_file is not None:
                resources = load_resources_from_bytes(uploaded_file.name, uploaded_file.getvalue(), max_pois=max_pois)
            else:
                resources = load_resources_from_path(
                    str(DEFAULT_OSM_PATH),
                    DEFAULT_OSM_PATH.stat().st_mtime,
                    max_pois=max_pois,
                    whitelist_path_str=str(DEFAULT_POI_WHITELIST_PATH),
                    whitelist_modified_time=DEFAULT_POI_WHITELIST_PATH.stat().st_mtime
                    if DEFAULT_POI_WHITELIST_PATH.exists()
                    else 0.0,
                )
    except Exception as exc:
        render_business_metrics(None)
        st.error(f"Could not load the OSM map: {exc}")
        return

    graph = resources["graph"]
    nodes = resources["nodes"]
    pois = order_pois_for_routing(graph, resources["pois"])
    metadata = resources["metadata"]

    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        render_business_metrics(None)
        st.error("The OSM file did not produce a usable routing graph. Check that it contains walkable campus roads.")
        return
    if len(pois) < 2:
        render_business_metrics(None)
        st.error("The map does not contain enough route locations. Try a larger OSM export.")
        return

    with st.sidebar:
        st.success(f"Map loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges, {len(pois):,} locations.")
        st.caption(metadata.get("source_label", "OSM source ready"))
        st.caption(f"POI source: {metadata.get('poi_source', 'Automatic OSM extraction')}")
        start_poi = st.selectbox("Start location", pois, format_func=format_poi_option, index=0)
        delivery_pois = selected_delivery_pois(start_poi, pois)
        generate_clicked = st.button("Generate Multi-stop Route", type="primary")

    validation_message = validate_delivery_selection(start_poi, delivery_pois)
    selection_key = route_selection_key(
        metadata,
        start_poi,
        delivery_pois,
        algorithm_name,
        prefer_footways,
        (departure_hour, departure_weekday) if congestion_model is not None and congestion_model.trained else None,
    )

    if generate_clicked:
        if validation_message:
            st.session_state["multi_route_result"] = {
                **build_preview_result(start_poi, delivery_pois, algorithm_name),
                "selection_key": selection_key,
                "message": validation_message,
            }
        else:
            with st.spinner("Calculating multi-stop route..."):
                route_graph = apply_route_preferences(graph, prefer_footways=prefer_footways)
                if congestion_model is not None and congestion_model.trained:
                    baseline_result = plan_multi_stop_route(
                        route_graph,
                        nodes,
                        start_poi,
                        delivery_pois,
                        algorithm_name,
                        resources.get("positions"),
                        max_exact_stops=MAX_DELIVERY_POINTS,
                    )
                    trained_graph = congestion_model.apply_time_of_day_costs(route_graph, departure_hour, departure_weekday)
                    result = plan_multi_stop_route(
                        trained_graph,
                        nodes,
                        start_poi,
                        delivery_pois,
                        algorithm_name,
                        resources.get("positions"),
                        max_exact_stops=MAX_DELIVERY_POINTS,
                    )
                    if result.get("success") and baseline_result.get("success"):
                        result["baseline_route"] = baseline_result
                        result["baseline_predicted_route_time"] = float(
                            congestion_model.predict_path_travel_time(
                                route_graph,
                                baseline_result.get("full_path", []),
                                departure_hour,
                                departure_weekday,
                            )
                        )
                        result["trained_predicted_route_time"] = float(result.get("total_cost", 0.0))
                else:
                    result = plan_multi_stop_route(
                        route_graph,
                        nodes,
                        start_poi,
                        delivery_pois,
                        algorithm_name,
                        resources.get("positions"),
                        max_exact_stops=MAX_DELIVERY_POINTS,
                    )
                result["selection_key"] = selection_key
                st.session_state["multi_route_result"] = result

    stored_result = st.session_state.get("multi_route_result")
    active_result = stored_result if stored_result and stored_result.get("selection_key") == selection_key else None

    render_business_metrics(active_result)

    preview_result = active_result or build_preview_result(start_poi, delivery_pois, algorithm_name)
    poi_lookup = {str(poi.get("nearest_graph_node", "")): poi for poi in [start_poi, *delivery_pois]}
    campus_map = render_multi_stop_map(
        graph,
        preview_result,
        poi_lookup,
        show_search_process=show_search_process,
        search_progress=search_progress,
        final_route_only=final_route_only,
        max_explored_edges=max_explored_edges,
        congestion_model=congestion_model,
        departure_hour=departure_hour,
        departure_weekday=departure_weekday,
        show_congestion_overlay=show_congestion_overlay,
    )
    st_folium(campus_map, height=620, width=None, returned_objects=[])

    if active_result:
        if active_result.get("success"):
            st.success(active_result["message"])
            render_delivery_order(active_result)

            st.markdown("### Technical Metrics")
            render_technical_metrics(active_result, algorithm_name)

            st.markdown("### Segment Details")
            st.dataframe(build_segment_table(active_result), width="stretch", hide_index=True)

            with st.expander("Full route node sequence", expanded=False):
                route_table = build_route_table(graph, active_result["full_path"])
                st.dataframe(route_table, width="stretch", hide_index=True, height=320)
                st.code(" -> ".join(str(node) for node in active_result["full_path"]))
        else:
            st.warning(active_result.get("message", "No feasible multi-stop route found."))
            st.markdown("### Technical Metrics")
            render_technical_metrics(None, algorithm_name)
    else:
        st.info("Choose a start location, add one or more delivery points, then generate a multi-stop route.")
        st.markdown("### Technical Metrics")
        render_technical_metrics(None, algorithm_name)

    render_algorithm_explanation(algorithm_name)

    st.markdown(
        f"""
        <div class="section-panel">
            <div class="inline-heading"><span>#</span> Graph Build Details</div>
            <p>Parsed nodes: {int(metadata.get("node_count", 0)):,} | Parsed ways: {int(metadata.get("way_count", 0)):,} | Routing edges skipped for missing nodes: {int(metadata.get("skipped_missing_edges", 0)):,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
