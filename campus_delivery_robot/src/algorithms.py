"""Search algorithms implemented without NetworkX shortest-path helpers."""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from typing import Any

from .utils import calculate_path_cost, calculate_path_distance, haversine_distance


MAX_EXPLORED_EDGES = 10_000


def _result(
    path: list[str],
    total_cost: float,
    total_distance: float,
    expanded_nodes: int,
    success: bool,
    message: str,
    explored_edges: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Create a consistent algorithm result dictionary."""
    return {
        "path": path,
        "total_cost": total_cost,
        "total_distance": total_distance,
        "expanded_nodes": expanded_nodes,
        "explored_edges": explored_edges or [],
        "success": success,
        "message": message,
    }


def _validate_nodes(graph: Any, start: str, goal: str) -> str | None:
    """Return an error message if start or goal is invalid."""
    if start not in graph:
        return "Start location is not connected to the routing graph."
    if goal not in graph:
        return "Goal location is not connected to the routing graph."
    return None


def _reconstruct_path(parent: dict[str, str | None], goal: str) -> list[str]:
    """Reconstruct a path from a parent pointer dictionary."""
    path = [goal]
    current = goal
    while parent[current] is not None:
        current = parent[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _record_explored_edge(
    explored_edges: list[tuple[str, str]],
    seen_edges: set[tuple[str, str]],
    u: str,
    v: str,
    max_edges: int = MAX_EXPLORED_EDGES,
) -> None:
    """Record a unique explored edge while keeping memory bounded."""
    if len(explored_edges) >= max_edges:
        return
    edge = (str(u), str(v))
    reverse_edge = (edge[1], edge[0])
    if edge in seen_edges or reverse_edge in seen_edges:
        return
    explored_edges.append(edge)
    seen_edges.add(edge)


def bfs_search(graph: Any, start: str, goal: str, max_expansions: int = 250_000) -> dict[str, Any]:
    """
    Breadth-first search.

    BFS ignores edge weights and finds a route with the fewest graph steps.
    """
    error = _validate_nodes(graph, start, goal)
    if error:
        return _result([], 0.0, 0.0, 0, False, error)
    if start == goal:
        return _result([start], 0.0, 0.0, 0, True, "Start and goal are the same node.")

    queue = deque([start])
    parent: dict[str, str | None] = {start: None}
    expanded = 0
    explored_edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    while queue:
        current = queue.popleft()
        expanded += 1

        if expanded > max_expansions:
            return _result(
                [],
                0.0,
                0.0,
                expanded,
                False,
                "Search stopped after reaching the expansion limit.",
                explored_edges,
            )

        if current == goal:
            path = _reconstruct_path(parent, goal)
            return _result(
                path,
                float(max(0, len(path) - 1)),
                calculate_path_distance(graph, path),
                expanded,
                True,
                "Route found with BFS.",
                explored_edges,
            )

        for neighbor in graph.neighbors(current):
            neighbor = str(neighbor)
            _record_explored_edge(explored_edges, seen_edges, current, neighbor)
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)

    return _result(
        [],
        0.0,
        0.0,
        expanded,
        False,
        "No route found between the selected locations.",
        explored_edges,
    )


def uniform_cost_search(
    graph: Any, start: str, goal: str, max_expansions: int = 250_000
) -> dict[str, Any]:
    """
    Uniform Cost Search.

    UCS expands the cheapest frontier node first and returns the lowest-cost
    route under the graph edge cost function.
    """
    error = _validate_nodes(graph, start, goal)
    if error:
        return _result([], 0.0, 0.0, 0, False, error)
    if start == goal:
        return _result([start], 0.0, 0.0, 0, True, "Start and goal are the same node.")

    tie_breaker = count()
    frontier: list[tuple[float, int, str]] = []
    heappush(frontier, (0.0, next(tie_breaker), start))
    best_cost: dict[str, float] = {start: 0.0}
    parent: dict[str, str | None] = {start: None}
    expanded_nodes: set[str] = set()
    explored_edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    while frontier:
        current_cost, _, current = heappop(frontier)
        if current in expanded_nodes:
            continue

        expanded_nodes.add(current)
        if len(expanded_nodes) > max_expansions:
            return _result(
                [],
                0.0,
                0.0,
                len(expanded_nodes),
                False,
                "Search stopped after reaching the expansion limit.",
                explored_edges,
            )

        if current == goal:
            path = _reconstruct_path(parent, goal)
            return _result(
                path,
                current_cost,
                calculate_path_distance(graph, path),
                len(expanded_nodes),
                True,
                "Route found with UCS.",
                explored_edges,
            )

        for neighbor in graph.neighbors(current):
            neighbor = str(neighbor)
            edge_cost = float(graph[current][neighbor].get("cost", graph[current][neighbor].get("distance", 1.0)))
            new_cost = current_cost + edge_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                _record_explored_edge(explored_edges, seen_edges, current, neighbor)
                best_cost[neighbor] = new_cost
                parent[neighbor] = current
                heappush(frontier, (new_cost, next(tie_breaker), neighbor))

    return _result(
        [],
        0.0,
        0.0,
        len(expanded_nodes),
        False,
        "No route found between the selected locations.",
        explored_edges,
    )


def _heuristic(graph: Any, node: str, goal: str) -> float:
    """Straight-line Haversine distance from a node to the goal."""
    node_data = graph.nodes[node]
    goal_data = graph.nodes[goal]
    return haversine_distance(
        float(node_data["lat"]),
        float(node_data["lon"]),
        float(goal_data["lat"]),
        float(goal_data["lon"]),
    )


def astar_search(graph: Any, start: str, goal: str, max_expansions: int = 250_000) -> dict[str, Any]:
    """
    A* search with Haversine straight-line distance as the heuristic.

    The heuristic guides the search toward the goal while edge costs still come
    from the weighted road-cost model.
    """
    error = _validate_nodes(graph, start, goal)
    if error:
        return _result([], 0.0, 0.0, 0, False, error)
    if start == goal:
        return _result([start], 0.0, 0.0, 0, True, "Start and goal are the same node.")

    tie_breaker = count()
    frontier: list[tuple[float, int, str]] = []
    heappush(frontier, (_heuristic(graph, start, goal), next(tie_breaker), start))
    best_g: dict[str, float] = {start: 0.0}
    parent: dict[str, str | None] = {start: None}
    expanded_nodes: set[str] = set()
    explored_edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    while frontier:
        _, _, current = heappop(frontier)
        if current in expanded_nodes:
            continue

        expanded_nodes.add(current)
        if len(expanded_nodes) > max_expansions:
            return _result(
                [],
                0.0,
                0.0,
                len(expanded_nodes),
                False,
                "Search stopped after reaching the expansion limit.",
                explored_edges,
            )

        if current == goal:
            path = _reconstruct_path(parent, goal)
            return _result(
                path,
                best_g[current],
                calculate_path_distance(graph, path),
                len(expanded_nodes),
                True,
                "Route found with A*.",
                explored_edges,
            )

        for neighbor in graph.neighbors(current):
            neighbor = str(neighbor)
            edge_cost = float(graph[current][neighbor].get("cost", graph[current][neighbor].get("distance", 1.0)))
            tentative_g = best_g[current] + edge_cost
            if tentative_g < best_g.get(neighbor, float("inf")):
                _record_explored_edge(explored_edges, seen_edges, current, neighbor)
                best_g[neighbor] = tentative_g
                parent[neighbor] = current
                f_score = tentative_g + _heuristic(graph, neighbor, goal)
                heappush(frontier, (f_score, next(tie_breaker), neighbor))

    return _result(
        [],
        0.0,
        0.0,
        len(expanded_nodes),
        False,
        "No route found between the selected locations.",
        explored_edges,
    )


ALGORITHMS = {
    "BFS": bfs_search,
    "Uniform Cost Search": uniform_cost_search,
    "A* Search": astar_search,
}
