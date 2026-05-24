"""Build a routable NetworkX graph and POI list from parsed OSM data."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .utils import haversine_distance


ALLOWED_HIGHWAYS = {
    "footway",
    "path",
    "pedestrian",
    "service",
    "residential",
    "living_street",
    "unclassified",
}

ROAD_TYPE_MULTIPLIERS = {
    "footway": 1.0,
    "pedestrian": 1.0,
    "path": 1.1,
    "living_street": 1.2,
    "service": 1.3,
    "residential": 1.4,
    "unclassified": 1.5,
    "steps": 9999.0,
}

POI_WAY_KEYS = {
    "building",
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "highway",
}


def calculate_edge_cost(distance_m: float, highway_type: str | None) -> float:
    """
    Convert physical distance into route cost.

    This function is deliberately isolated so future versions can add battery,
    crowd, slope, weather, or delivery-priority penalties.
    """
    multiplier = ROAD_TYPE_MULTIPLIERS.get(highway_type or "", 2.0)
    return distance_m * multiplier


def _preferred_name(tags: dict[str, str]) -> str | None:
    """Pick the most presentation-friendly name available in OSM tags."""
    return (
        tags.get("name:en")
        or tags.get("official_name:en")
        or tags.get("short_name:en")
        or tags.get("name")
        or tags.get("official_name")
        or tags.get("short_name")
    )


def _add_graph_node(graph: nx.Graph, node: dict[str, Any]) -> None:
    """Add a node with coordinate attributes if it is not already present."""
    node_id = str(node["id"])
    if node_id in graph:
        return

    tags = node.get("tags", {})
    graph.add_node(
        node_id,
        osm_id=node_id,
        lat=float(node["lat"]),
        lon=float(node["lon"]),
        name=_preferred_name(tags),
    )


def build_graph(nodes: dict[str, dict[str, Any]], ways: list[dict[str, Any]]) -> nx.Graph:
    """
    Build an undirected campus routing graph from walkable OSM ways.

    Consecutive node references in each allowed way become graph edges. Ways
    with private access or steps are skipped for the MVP.
    """
    graph = nx.Graph()
    skipped_missing_edges = 0
    skipped_private_ways = 0

    for way in ways:
        tags = way.get("tags", {})
        highway_type = tags.get("highway")
        if highway_type == "steps" or highway_type not in ALLOWED_HIGHWAYS:
            continue
        if tags.get("access") == "private":
            skipped_private_ways += 1
            continue

        node_refs = [str(ref) for ref in way.get("node_refs", [])]
        if len(node_refs) < 2:
            continue

        road_name = _preferred_name(tags)
        for u, v in zip(node_refs, node_refs[1:]):
            if u == v:
                continue
            node_u = nodes.get(u)
            node_v = nodes.get(v)
            if not node_u or not node_v:
                skipped_missing_edges += 1
                continue

            distance = haversine_distance(
                float(node_u["lat"]),
                float(node_u["lon"]),
                float(node_v["lat"]),
                float(node_v["lon"]),
            )
            if distance <= 0:
                continue

            base_cost = calculate_edge_cost(distance, highway_type)
            _add_graph_node(graph, node_u)
            _add_graph_node(graph, node_v)

            edge_data = {
                "distance": distance,
                "base_cost": base_cost,
                "cost": base_cost,
                "highway": highway_type,
                "name": road_name,
                "way_id": str(way["id"]),
                "barrier": tags.get("barrier"),
            }

            if graph.has_edge(u, v):
                if base_cost < float(graph[u][v].get("cost", float("inf"))):
                    graph[u][v].update(edge_data)
            else:
                graph.add_edge(u, v, **edge_data)

    graph.graph["skipped_missing_edges"] = skipped_missing_edges
    graph.graph["skipped_private_ways"] = skipped_private_ways
    graph.graph["allowed_highways"] = sorted(ALLOWED_HIGHWAYS)
    return graph


def apply_route_preferences(
    graph: nx.Graph,
    avoid_service_roads: bool = False,
    prefer_footways: bool = False,
) -> nx.Graph:
    """Return a graph copy with user-selected route preference penalties."""
    adjusted = graph.copy()

    for _, _, data in adjusted.edges(data=True):
        highway = data.get("highway")
        cost = float(data.get("base_cost", data.get("cost", data.get("distance", 1.0))))

        if avoid_service_roads and highway == "service":
            cost *= 2.5
        if prefer_footways and highway not in {"footway", "pedestrian"}:
            cost *= 1.15

        data["cost"] = cost

    return adjusted


def find_nearest_graph_node(
    lat: float,
    lon: float,
    graph: nx.Graph,
    graph_positions: list[tuple[str, float, float]] | None = None,
) -> tuple[str | None, float]:
    """Find the nearest routing graph node to a latitude/longitude point."""
    positions = graph_positions
    if positions is None:
        positions = [
            (str(node_id), float(data["lat"]), float(data["lon"]))
            for node_id, data in graph.nodes(data=True)
            if data.get("lat") is not None and data.get("lon") is not None
        ]

    best_node = None
    best_distance = float("inf")
    for node_id, node_lat, node_lon in positions:
        distance = haversine_distance(lat, lon, node_lat, node_lon)
        if distance < best_distance:
            best_node = node_id
            best_distance = distance

    return best_node, best_distance


def _way_centroid(
    way: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> tuple[float, float] | None:
    """Approximate a way centroid by averaging available referenced nodes."""
    lat_sum = 0.0
    lon_sum = 0.0
    count = 0
    for ref in way.get("node_refs", []):
        node = nodes.get(str(ref))
        if not node:
            continue
        lat_sum += float(node["lat"])
        lon_sum += float(node["lon"])
        count += 1

    if count == 0:
        return None
    return lat_sum / count, lon_sum / count


def _is_named_poi_way(tags: dict[str, str]) -> bool:
    """Return True for named ways useful as route choices."""
    return bool(_preferred_name(tags)) and any(key in tags for key in POI_WAY_KEYS)


def _make_unique_display_names(pois: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make POI display labels unique while preserving the original order."""
    counts: dict[str, int] = {}
    for poi in pois:
        base = str(poi["display_name"])
        counts[base] = counts.get(base, 0) + 1
        if counts[base] > 1:
            poi["display_name"] = f"{base} ({counts[base]})"
    return pois


def extract_pois(
    nodes: dict[str, dict[str, Any]],
    ways: list[dict[str, Any]],
    graph: nx.Graph,
    max_pois: int = 150,
    min_named_pois: int = 10,
    fallback_count: int = 40,
) -> list[dict[str, Any]]:
    """
    Extract named POIs and snap each one to the nearest routing graph node.

    If the OSM file has too few named places, uniformly sampled graph nodes are
    appended so the Streamlit app remains usable for any valid map.
    """
    if graph.number_of_nodes() == 0:
        return []

    graph_positions = [
        (str(node_id), float(data["lat"]), float(data["lon"]))
        for node_id, data in graph.nodes(data=True)
        if data.get("lat") is not None and data.get("lon") is not None
    ]

    candidates: list[dict[str, Any]] = []

    for node_id, node in nodes.items():
        tags = node.get("tags", {})
        name = _preferred_name(tags)
        if not name:
            continue
        candidates.append(
            {
                "priority": 0,
                "display_name": name,
                "osm_id": str(node_id),
                "lat": float(node["lat"]),
                "lon": float(node["lon"]),
                "source": "named_node",
            }
        )

    for way in ways:
        tags = way.get("tags", {})
        if not _is_named_poi_way(tags):
            continue
        centroid = _way_centroid(way, nodes)
        if centroid is None:
            continue
        name = _preferred_name(tags)
        category = next((key for key in POI_WAY_KEYS if key in tags), "way")
        candidates.append(
            {
                "priority": 1 if category != "highway" else 2,
                "display_name": str(name),
                "osm_id": str(way["id"]),
                "lat": centroid[0],
                "lon": centroid[1],
                "source": f"named_{category}",
            }
        )

    candidates.sort(key=lambda item: (item["priority"], str(item["display_name"]).lower()))

    pois: list[dict[str, Any]] = []
    seen_identity = set()
    used_nearest_nodes = set()

    for candidate in candidates:
        if len(pois) >= max_pois:
            break

        identity = (
            str(candidate["display_name"]).strip().lower(),
            round(float(candidate["lat"]), 6),
            round(float(candidate["lon"]), 6),
        )
        if identity in seen_identity:
            continue
        seen_identity.add(identity)

        nearest_node, nearest_distance = find_nearest_graph_node(
            float(candidate["lat"]),
            float(candidate["lon"]),
            graph,
            graph_positions,
        )
        if nearest_node is None or nearest_node in used_nearest_nodes:
            continue

        candidate["nearest_graph_node"] = nearest_node
        candidate["nearest_distance_m"] = nearest_distance
        pois.append(candidate)
        used_nearest_nodes.add(nearest_node)

    if len(pois) < min_named_pois:
        graph_nodes = list(graph.nodes(data=True))
        target_count = min(fallback_count, max_pois)
        step = max(1, len(graph_nodes) // max(target_count, 1))
        fallback_index = 1

        for node_id, data in graph_nodes[::step]:
            if len(pois) >= target_count:
                break
            node_id = str(node_id)
            if node_id in used_nearest_nodes:
                continue
            pois.append(
                {
                    "display_name": f"Map Node {fallback_index}",
                    "osm_id": node_id,
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "nearest_graph_node": node_id,
                    "nearest_distance_m": 0.0,
                    "source": "fallback_graph_node",
                }
            )
            used_nearest_nodes.add(node_id)
            fallback_index += 1

    return _make_unique_display_names(pois[:max_pois])

