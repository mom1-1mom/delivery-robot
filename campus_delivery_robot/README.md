# Campus Delivery Robot Path Planning

This project is a runnable MVP for an AI search project: **Campus Delivery Robot Path Planning in Central South University**. It reads local OpenStreetMap `.osm` data, builds a walkable campus road graph, lets users choose start and goal locations, runs a selected search algorithm, and visualizes the resulting robot delivery route on a real map.

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
- Shows route metrics:
  - total distance
  - total cost
  - nodes expanded
  - running time
  - full node sequence
- Visualizes the route with Folium inside Streamlit.
- Caches parsing and graph building for smoother interaction.

## Installation

Create and activate a Python environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
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

## Data File

Place your OpenStreetMap export here:

```text
data/campus.osm
```

This repository is prepared to use `data/campus.osm` by default. You can also upload another `.osm` file from the sidebar in the web app.

## Algorithms

### BFS

Breadth-First Search ignores edge weights. It finds a route with the fewest graph steps, which may not be the shortest physical route.

### Uniform Cost Search

Uniform Cost Search uses the weighted edge cost. It expands the lowest cumulative-cost frontier node first and returns the lowest-cost route under the cost model.

### A* Search

A* Search uses:

```text
f(n) = g(n) + h(n)
```

- `g(n)`: accumulated weighted route cost
- `h(n)`: straight-line Haversine distance from the current node to the goal

A* is usually faster than UCS because the heuristic guides the search toward the goal.

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
|   |-- map_renderer.py
|   |-- utils.py
```

## Common Questions

### The app says no OSM file was found.

Put your map file at `data/campus.osm`, or upload an `.osm` file in the sidebar.

### There are not many named locations.

Some OSM exports have few named POIs. The app automatically adds sampled graph nodes named `Map Node 1`, `Map Node 2`, and so on.

### The selected start and goal cannot be routed.

They may belong to disconnected parts of the map graph. Choose two locations closer to the same campus road network.

### BFS returns a strange route.

That is expected. BFS minimizes graph steps, not meters or robot travel cost. Use UCS or A* for weighted routing.

### A way references a missing node.

The graph builder skips that edge and continues, so incomplete OSM exports should not crash the app.

