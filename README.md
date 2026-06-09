# Campus Delivery Robot Route Planner

This project is a Streamlit-based campus delivery routing MVP for Central South University. It parses local OpenStreetMap data, builds a walkable road graph, plans optimized multi-stop deliveries, predicts time-dependent edge travel times with machine learning, and visualizes the final route on an interactive Folium map.

## Main Features

- Parses local `.osm` XML data.
- Builds a NetworkX graph from walkable campus roads.
- Uses Haversine distance to calculate edge lengths in meters.
- Stores distance, road type, static cost, and predicted travel time on graph edges.
- Uses NetworkX for graph storage and connectivity only.
- Implements BFS, Uniform Cost Search, and A* manually.
- Supports one start point and up to eight delivery points.
- Optimizes delivery order using pairwise route searches and exact permutation search.
- Loads cleaned English POI names from `data/poi_whitelist.csv`.
- Displays the start marker, numbered delivery stops, route segments, final route, and optional congestion overlay.
- Trains a `RandomForestRegressor` from uploaded historical congestion data.
- Persists the latest valid congestion CSV across browser refreshes.
- Uses batch edge prediction and cached graph costs for responsive ML routing.
- Displays business and technical route metrics.

The former search-process visualization controls and explored-edge map layer have been removed from the current UI.

## Installation

From the repository root:

```powershell
cd campus_delivery_robot
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
```

Required packages are listed in `campus_delivery_robot/requirements.txt`:

- Streamlit
- streamlit-folium
- Folium
- NetworkX
- Pandas
- scikit-learn

## Run

Run the application from `campus_delivery_robot`:

```powershell
.\.venv-win\Scripts\streamlit.exe run app.py
```

Then open:

```text
http://localhost:8501
```

The included `.streamlit/config.toml` disables Streamlit usage-stat collection and runs the server in headless mode.

## Data Files

```text
campus_delivery_robot/data/
|-- campus.osm
|-- poi_whitelist.csv
|-- sample_congestion.csv
`-- uploaded_congestion.csv
```

- `campus.osm`: default local campus map.
- `poi_whitelist.csv`: manually cleaned POIs used by the location selectors.
- `sample_congestion.csv`: example historical traffic dataset.
- `uploaded_congestion.csv`: latest valid uploaded training dataset. It is created automatically and excluded from Git.

The application currently uses `data/campus.osm` as its map source.

## Route Planning

1. Select BFS, Uniform Cost Search, or A*.
2. Optionally enable `Prefer footways`.
3. Select a start location.
4. Add one or more delivery points.
5. Optionally upload a congestion CSV and select the departure hour and weekday.
6. Click `Generate Multi-stop Route`.

The planner first computes routes between every important POI pair:

```text
important_points = [start] + delivery_points
```

It then evaluates delivery-point permutations and chooses the order with the lowest combined objective. Finally, the selected route segments are joined into one complete delivery path.

## Search Algorithms

### BFS

BFS uses a FIFO queue and finds a route with the fewest graph edges. It ignores weighted distance and ML travel-time costs.

### Uniform Cost Search

UCS expands the frontier node with the lowest accumulated edge cost and returns the lowest-cost route under the active cost model.

### A* Search

A* uses:

```text
f(n) = g(n) + h(n)
```

- `g(n)`: accumulated graph edge cost.
- `h(n)`: straight-line Haversine distance to the destination.

## Static Cost Model

Each edge initially uses:

```text
edge_cost = distance_m * road_type_multiplier
```

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

When `Prefer footways` is enabled, non-footway and non-pedestrian edges receive an additional `1.15` cost multiplier.

## Traffic Congestion Model

The ML module uses a Random Forest regressor with these features:

- edge distance
- hour
- weekday
- highway type
- edge identifier

The target column is edge travel time in seconds. Accepted target names include `travel_time`, `travel_time_s`, `duration`, and similar variants.

After training:

```text
edge.cost = predicted_travel_time_seconds
```

UCS and A* therefore optimize predicted travel time when the model is active. Predictions for all graph edges are generated in one batch and stored on the adjusted graph.

After a valid CSV is uploaded, it is saved as `data/uploaded_congestion.csv`. Refreshing the browser automatically restores and retrains the model from this file.

## Metrics and Formulas

### Total Distance

```text
total_distance = sum(edge_distance)
```

### Total Cost

Without ML, UCS and A* use:

```text
total_cost = sum(distance * road_type_multiplier)
```

With ML, UCS and A* use:

```text
total_cost = sum(predicted_edge_travel_time_seconds)
```

For BFS:

```text
total_cost = number_of_graph_edges
```

### Estimated Delivery Time

Without ML:

```text
estimated_minutes = weighted_distance / 1.2 / 60
```

With ML:

```text
estimated_minutes = predicted_seconds * 0.5 / 60
```

The current ML display calibration factor is `0.5`.

### Predicted Delivery Fee

The fee is a simulated rule-based value, not an ML prediction:

```text
predicted_fee = 2.0 + total_distance_km * 1.5 + delivery_count * 0.5
```

## POI Cleaning

The application prioritizes `data/poi_whitelist.csv` instead of raw OSM names. The CSV supports:

- `enabled`: whether the POI appears in the application.
- `display_name`: cleaned English name.
- `original_name`: original OSM name.
- `nearest_graph_node`: routing node used by the planner.
- `component_size`: connected-component size used during cleaning.

`export_pois.py` is an offline helper for regenerating the whitelist:

```powershell
.\.venv-win\Scripts\python.exe export_pois.py --overwrite
```

The web application does not require this script during normal operation.

## Project Structure

```text
AI project/
|-- README.md
|-- TRAFFIC_MODEL_GUIDE.md
|-- example_traffic_model.py
`-- campus_delivery_robot/
    |-- app.py
    |-- export_pois.py
    |-- requirements.txt
    |-- .streamlit/
    |   `-- config.toml
    |-- data/
    |   |-- campus.osm
    |   |-- poi_whitelist.csv
    |   `-- sample_congestion.csv
    `-- src/
        |-- __init__.py
        |-- osm_parser.py
        |-- graph_builder.py
        |-- algorithms.py
        |-- multistop_planner.py
        |-- map_renderer.py
        |-- congestion_model.py
        `-- utils.py
```

## Limitations

- Route quality depends on the completeness of the OSM map and POI whitelist.
- The congestion model is only as reliable as its uploaded historical data.
- The delivery fee is a fixed simulation formula.
- Exact permutation search grows factorially with the number of delivery points.
- Weather, crowd density, battery status, elevation, stairs, and real-time obstacles are not yet included.
- A larger city-scale deployment would require more scalable routing and delivery-order optimization methods.
