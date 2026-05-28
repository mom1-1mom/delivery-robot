# Campus Delivery Robot Route Planner

This project is a runnable MVP for **Campus Delivery Robot Path Planning in Central South University**. It reads local OpenStreetMap `.osm` data, builds a walkable campus road graph, lets users choose one start location and multiple delivery points, plans an optimized multi-stop delivery order, and visualizes the final route plus optional search-process edges on a real map.

## Features

- Parses local OpenStreetMap `.osm` XML data.
- Builds a NetworkX graph from campus-friendly roads:
  - `footway`
  - `path`
  - `pedestrian`
  - `service`
  - `residential`
  - `living_street`
  - `unclassified`
- Skips `steps` and `access=private` ways for the MVP.
- Converts consecutive OSM way nodes into weighted graph edges.
- Uses Haversine distance in meters.
- Applies a road-type cost multiplier for robot routing.
- Extracts named POIs from OSM nodes and ways.
- Falls back to sampled graph nodes if the map has too few named POIs.
- Implements BFS, Uniform Cost Search, and A* Search manually.
- Supports multi-stop delivery route planning for up to 8 delivery points.
- Optimizes delivery order using exact permutation search.
- Shows search-process visualization with explored edges.
- Displays business metrics first: estimated time, total distance, and delivery fee.
- Displays technical metrics: total cost, running time, nodes expanded, algorithm, and route node count.
- Visualizes start, numbered delivery stops, route segments, and final route with Folium inside Streamlit.
- Caches parsing and graph building for smoother interaction.

## Installation

Create and activate a Python environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

If your system Python blocks global `pip install`, use a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

From this project folder, run:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

The project includes `.streamlit/config.toml` to disable Streamlit onboarding prompts and usage-stat collection for easier classroom demos.

## Data File

Place your OpenStreetMap export here:

```text
data/campus.osm
```

This repository is prepared to use `data/campus.osm` by default. You can also upload another `.osm` file from the sidebar in the web app.

## How To Plan A Multi-stop Delivery Route

1. Choose a `Start location`.
2. Choose one or more `Delivery point` entries in the sidebar.
3. Click `+ Add Stop` to add another delivery point.
4. Select `BFS`, `Uniform Cost Search`, or `A* Search`.
5. Choose whether to show the search process.
6. Click `Generate Multi-stop Route`.

The app automatically decides the best delivery order and then joins all route segments into one full delivery route.

## Multi-stop Planning Method

The planner uses two layers:

### 1. Pairwise Route Calculation

For all important points:

```text
important_points = [start] + delivery_points
```

the selected search algorithm computes routes such as:

```text
start -> A
start -> B
A -> B
B -> C
```

Each pairwise search returns:

- path
- distance
- cost
- expanded nodes
- explored edges

### 2. Delivery Order Optimization

For up to 8 delivery points, the planner tries every possible delivery order and selects the one with the lowest total route objective.

- In `BFS` mode, the order mainly follows the fewest graph steps.
- In `UCS` mode, the order minimizes weighted route cost.
- In `A*` mode, the order uses the same weighted cost, while A* speeds up pairwise search with a straight-line heuristic.

## Algorithms

### BFS

Breadth-First Search ignores edge weights. It finds a route with the fewest graph steps, which may not be the shortest physical route or cheapest robot route.

### Uniform Cost Search

Uniform Cost Search uses weighted edge cost. It expands the lowest cumulative-cost frontier node first and returns the lowest-cost route under the road-cost model.

### A* Search

A* Search uses:

```text
f(n) = g(n) + h(n)
```

- `g(n)`: accumulated weighted route cost
- `h(n)`: straight-line Haversine distance from the current node to the goal

A* usually expands fewer nodes than UCS because the heuristic guides the search toward the goal.

## Search Process Visualization

Each algorithm returns `explored_edges`, which records graph edges touched during search. In the app:

- explored edges are shown as light blue lines
- the current progress batch is shown in orange
- the final delivery route is highlighted in green
- numbered stop markers show the optimized delivery sequence

Use the `Search progress` slider to inspect the search process in stages. Use `Final route only` to hide explored edges and keep the map focused on the final delivery route.

## Metric Definitions

| Metric | Meaning |
|---|---|
| Total Distance | Real route length in meters, calculated from OSM coordinates. |
| Estimated Delivery Time | Distance-based delivery time using a default robot speed of 1.2 m/s and small road-type time penalties. |
| Delivery Fee | Simulated CNY fee: base fee + distance fee + stop fee. |
| Total Cost | Internal algorithm objective based on edge weights. For BFS, this is graph-step oriented. |
| Running Time | Real wall-clock time used by the planner. |
| Nodes Expanded | Number of graph nodes expanded by the selected search algorithm across all selected route segments. |

## Cost Model

Each graph edge stores:

- `distance`
- `base_cost`
- `cost`
- `highway`
- `name`

Default cost:

```text
cost = distance * road_type_multiplier
```

Multipliers:

| Road type | Multiplier |
|---|---:|
| footway | 1.0 |
| pedestrian | 1.0 |
| path | 1.1 |
| living_street | 1.2 |
| service | 1.3 |
| residential | 1.4 |
| unclassified | 1.5 |
| steps | excluded |

The cost function is isolated in `src/graph_builder.py`, so later versions can add battery cost, crowd penalty, terrain slope, weather, or delivery urgency.

## Project Structure

```text
campus_delivery_robot/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- data/
|   |-- campus.osm
|
|-- src/
|   |-- __init__.py
|   |-- osm_parser.py
|   |-- graph_builder.py
|   |-- algorithms.py
|   |-- multistop_planner.py
|   |-- map_renderer.py
|   |-- utils.py
```

## Common Questions

### The app says no OSM file was found.

Put your map file at `data/campus.osm`, or upload an `.osm` file in the sidebar.

### There are not many named locations.

Some OSM exports have few named POIs. The app automatically adds sampled graph nodes named `Map Node 1`, `Map Node 2`, and so on.

### A delivery point cannot be routed.

It may belong to a disconnected part of the map graph. Choose delivery points closer to the same campus road network.

### BFS returns a strange route.

That is expected. BFS minimizes graph steps, not meters or weighted robot travel cost. Use UCS or A* for weighted routing.

### Search-process visualization feels slow.

Reduce `Max explored edges` in the sidebar or enable `Final route only`.

### A way references a missing node.

The graph builder skips that edge and continues, so incomplete OSM exports should not crash the app.

