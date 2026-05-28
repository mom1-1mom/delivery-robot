"""Multi-stop delivery route planning built on pairwise graph search."""

from __future__ import annotations

from itertools import permutations
from time import perf_counter
from typing import Any

from .algorithms import ALGORITHMS
from .utils import calculate_path_cost, calculate_path_distance


ROBOT_SPEED_MPS = 1.2
MAX_COLLECTED_EXPLORED_EDGES = 20_000

ROAD_TIME_MULTIPLIERS = {
    "footway": 1.0,
    "pedestrian": 1.0,
    "path": 1.05,
    "living_street": 1.08,
    "service": 1.15,
    "residential": 1.12,
    "unclassified": 1.18,
}


def _failure(message: str, running_time: float = 0.0, explored_edges: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Return a consistent failed planning result."""
    return {
        "success": False,
        "delivery_order": [],
        "full_path": [],
        "segments": [],
        "total_distance": 0.0,
        "total_cost": 0.0,
        "estimated_time": 0.0,
        "delivery_fee": 0.0,
        "running_time": running_time,
        "total_expanded_nodes": 0,
        "all_explored_edges": explored_edges or [],
        "message": message,
    }


def _poi_node(poi: dict[str, Any]) -> str | None:
    """Return a POI's snapped graph node id."""
    node = poi.get("nearest_graph_node")
    return str(node) if node is not None and str(node) else None


def _poi_name(poi: dict[str, Any], fallback: str) -> str:
    """Return a display-safe POI name for summaries."""
    return str(poi.get("display_name") or fallback)


def _dedupe_edges(
    target: list[tuple[str, str]],
    seen: set[tuple[str, str]],
    edges: list[tuple[str, str]],
    limit: int = MAX_COLLECTED_EXPLORED_EDGES,
) -> None:
    """Append unique explored edges with a hard cap for UI performance."""
    for u, v in edges:
        if len(target) >= limit:
            return
        edge = (str(u), str(v))
        reverse = (edge[1], edge[0])
        if edge in seen or reverse in seen:
            continue
        target.append(edge)
        seen.add(edge)


def _pair_metric(route: dict[str, Any], algorithm_name: str) -> float:
    """Return the optimisation metric for a pairwise route."""
    if algorithm_name == "BFS":
        return float(max(0, len(route.get("path", [])) - 1))
    return float(route.get("total_cost", float("inf")))


def _calculate_estimated_time_minutes(graph: Any, path: list[str], robot_speed_mps: float = ROBOT_SPEED_MPS) -> float:
    """Estimate delivery time using distance, robot speed, and road-type penalties."""
    if not path or robot_speed_mps <= 0:
        return 0.0

    weighted_distance = 0.0
    for u, v in zip(path, path[1:]):
        if not graph.has_edge(u, v):
            continue
        edge_data = graph[u][v]
        highway = edge_data.get("highway", "unclassified")
        multiplier = ROAD_TIME_MULTIPLIERS.get(highway, 1.1)
        weighted_distance += float(edge_data.get("distance", 0.0)) * multiplier

    return weighted_distance / robot_speed_mps / 60


def _calculate_delivery_fee(total_distance_m: float, delivery_count: int) -> float:
    """Calculate a simple simulated CNY delivery fee."""
    base_fee = 2.0
    distance_fee = total_distance_m / 1000 * 1.5
    stop_fee = delivery_count * 0.5
    return base_fee + distance_fee + stop_fee


def _build_pairwise_routes(
    graph: Any,
    important_pois: list[dict[str, Any]],
    algorithm_name: str,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[tuple[str, str]]]:
    """Run the selected point-to-point algorithm between all important POIs."""
    search_fn = ALGORITHMS[algorithm_name]
    pairwise_routes: dict[tuple[int, int], dict[str, Any]] = {}
    all_explored_edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    for from_index, from_poi in enumerate(important_pois):
        from_node = _poi_node(from_poi)
        for to_index, to_poi in enumerate(important_pois):
            if from_index == to_index:
                continue
            to_node = _poi_node(to_poi)
            if from_node is None or to_node is None:
                continue

            route = search_fn(graph, from_node, to_node)
            route["from_index"] = from_index
            route["to_index"] = to_index
            route["optimization_cost"] = _pair_metric(route, algorithm_name)
            if route.get("success"):
                route["total_distance"] = float(route.get("total_distance", calculate_path_distance(graph, route["path"])))
                route["weighted_route_cost"] = calculate_path_cost(graph, route["path"])
            pairwise_routes[(from_index, to_index)] = route
            _dedupe_edges(all_explored_edges, seen_edges, route.get("explored_edges", []))

    return pairwise_routes, all_explored_edges


def _exact_delivery_order(
    delivery_indices: list[int],
    pairwise_routes: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[int], float] | None:
    """Find the lowest-cost delivery order by exact permutation search."""
    best_order: list[int] | None = None
    best_cost = float("inf")

    for order in permutations(delivery_indices):
        current = 0
        total = 0.0
        feasible = True
        for next_index in order:
            route = pairwise_routes.get((current, next_index))
            if not route or not route.get("success"):
                feasible = False
                break
            total += float(route["optimization_cost"])
            current = next_index

        if feasible and total < best_cost:
            best_cost = total
            best_order = list(order)

    if best_order is None:
        return None
    return best_order, best_cost


def _greedy_delivery_order(
    delivery_indices: list[int],
    pairwise_routes: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[int], float] | None:
    """Fallback nearest-neighbour order for larger delivery sets."""
    remaining = set(delivery_indices)
    order: list[int] = []
    current = 0
    total = 0.0

    while remaining:
        candidates = []
        for next_index in remaining:
            route = pairwise_routes.get((current, next_index))
            if route and route.get("success"):
                candidates.append((float(route["optimization_cost"]), next_index))
        if not candidates:
            return None
        step_cost, chosen = min(candidates, key=lambda item: item[0])
        total += step_cost
        order.append(chosen)
        remaining.remove(chosen)
        current = chosen

    return order, total


def _compose_segments(
    graph: Any,
    important_pois: list[dict[str, Any]],
    order: list[int],
    pairwise_routes: dict[tuple[int, int], dict[str, Any]],
    algorithm_name: str,
) -> tuple[list[dict[str, Any]], list[str], float, float, int]:
    """Build segment records and concatenate their paths without duplicate joins."""
    segments: list[dict[str, Any]] = []
    full_path: list[str] = []
    total_distance = 0.0
    total_cost = 0.0
    total_expanded = 0
    current = 0

    for stop_number, next_index in enumerate(order, start=1):
        route = pairwise_routes[(current, next_index)]
        path = [str(node_id) for node_id in route.get("path", [])]
        if not path:
            raise ValueError("A planned segment has an empty path.")

        if full_path and full_path[-1] == path[0]:
            full_path.extend(path[1:])
        else:
            full_path.extend(path)

        distance = float(route.get("total_distance", calculate_path_distance(graph, path)))
        cost = float(route.get("total_cost", 0.0))
        total_distance += distance
        total_cost += cost
        total_expanded += int(route.get("expanded_nodes", 0))

        from_poi = important_pois[current]
        to_poi = important_pois[next_index]
        segments.append(
            {
                "segment": stop_number,
                "from": "Start" if current == 0 else f"Stop {stop_number - 1}",
                "to": f"Stop {stop_number}",
                "from_name": _poi_name(from_poi, "Start"),
                "to_name": _poi_name(to_poi, f"Stop {stop_number}"),
                "from_poi": from_poi,
                "to_poi": to_poi,
                "path": path,
                "distance": distance,
                "cost": cost,
                "optimization_cost": float(route.get("optimization_cost", cost)),
                "expanded_nodes": int(route.get("expanded_nodes", 0)),
                "algorithm": algorithm_name,
            }
        )
        current = next_index

    return segments, full_path, total_distance, total_cost, total_expanded


def plan_multi_stop_route(
    graph: Any,
    nodes: dict[str, dict[str, Any]] | None,
    start_poi: dict[str, Any],
    delivery_pois: list[dict[str, Any]],
    algorithm_name: str,
    positions: list[tuple[str, float, float]] | None = None,
    max_exact_stops: int = 8,
) -> dict[str, Any]:
    """
    Plan a complete multi-stop delivery route.

    The planner first computes pairwise routes between the start and every
    delivery point using the selected search algorithm. It then optimises the
    delivery order by exact permutation search for up to `max_exact_stops`.
    """
    del nodes, positions
    start_time = perf_counter()

    if algorithm_name not in ALGORITHMS:
        return _failure(f"Unknown algorithm: {algorithm_name}")

    if not delivery_pois:
        return _failure("Select at least one delivery point.")
    start_node = _poi_node(start_poi)
    if start_node is None:
        return _failure("Start location does not have a nearest graph node.")
    if start_node not in graph:
        return _failure("Start location is not connected to the routing graph.")

    delivery_nodes: list[str] = []
    for index, poi in enumerate(delivery_pois, start=1):
        node = _poi_node(poi)
        if node is None:
            return _failure(f"Delivery point {index} does not have a nearest graph node.")
        if node == start_node:
            return _failure("Start location cannot also be a delivery point.")
        if node not in graph:
            return _failure(f"Delivery point {index} is not connected to the routing graph.")
        if node in delivery_nodes:
            return _failure("Two selected delivery points resolve to the same graph node.")
        delivery_nodes.append(node)

    important_pois = [start_poi] + delivery_pois
    pairwise_routes, all_explored_edges = _build_pairwise_routes(graph, important_pois, algorithm_name)

    delivery_indices = list(range(1, len(important_pois)))
    if len(delivery_indices) <= max_exact_stops:
        order_result = _exact_delivery_order(delivery_indices, pairwise_routes)
        planning_method = "Exact permutation search"
    else:
        order_result = _greedy_delivery_order(delivery_indices, pairwise_routes)
        planning_method = "Greedy nearest-neighbour heuristic"

    if order_result is None:
        return _failure(
            "No feasible multi-stop route found. At least one selected delivery point may be disconnected.",
            perf_counter() - start_time,
            all_explored_edges,
        )

    order, _ = order_result
    try:
        segments, full_path, total_distance, total_cost, total_expanded = _compose_segments(
            graph,
            important_pois,
            order,
            pairwise_routes,
            algorithm_name,
        )
    except ValueError as exc:
        return _failure(str(exc), perf_counter() - start_time, all_explored_edges)

    if not full_path:
        return _failure("The planned full path is empty.", perf_counter() - start_time, all_explored_edges)

    ordered_delivery_pois = [important_pois[index] for index in order]
    estimated_time = _calculate_estimated_time_minutes(graph, full_path)
    delivery_fee = _calculate_delivery_fee(total_distance, len(delivery_pois))

    return {
        "success": True,
        "delivery_order": ordered_delivery_pois,
        "full_path": full_path,
        "segments": segments,
        "total_distance": total_distance,
        "total_cost": total_cost,
        "estimated_time": estimated_time,
        "delivery_fee": delivery_fee,
        "running_time": perf_counter() - start_time,
        "total_expanded_nodes": total_expanded,
        "all_explored_edges": all_explored_edges,
        "start_poi": start_poi,
        "delivery_pois": delivery_pois,
        "algorithm": algorithm_name,
        "planning_method": planning_method,
        "message": "Multi-stop route planned successfully.",
    }
