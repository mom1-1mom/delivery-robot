"""Example: Using TrafficCongestionModel programmatically."""

from pathlib import Path

import pandas as pd

from src.congestion_model import TrafficCongestionModel
from src.graph_builder import build_graph
from src.osm_parser import parse_osm_file
from src.multistop_planner import plan_multi_stop_route

# Load OSM map
osm_path = Path("data/campus.osm")
nodes, ways, metadata = parse_osm_file(osm_path)
graph = build_graph(nodes, ways)

print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Load historical congestion data and train model
congestion_data = pd.read_csv("data/sample_congestion.csv")
model = TrafficCongestionModel()
model.train_from_dataframe(congestion_data)

print(f"\nModel trained:")
report = model.report()
print(f"  - Samples: {report['train_samples']}")
print(f"  - Validation RMSE: {report['train_rmse_seconds']:.1f} sec")

# Option 1: Predict travel time for a single edge at 8 AM
sample_edge = graph.edges(data=True).__iter__().__next__()[2]
predicted_time_8am = model.predict_edge_travel_time(sample_edge, hour=8)
print(f"\nEdge travel time at 8 AM: {predicted_time_8am:.1f} sec")

# Option 2: Predict travel time for a path
path_example = list(graph.nodes())[:5]  # First 5 nodes as example
path_time = model.predict_path_travel_time(graph, path_example, hour=14)
print(f"Path travel time at 2 PM: {path_time:.1f} sec")

# Option 3: Get congestion-adjusted graph for specific hour
adjusted_graph_8am = model.apply_time_of_day_costs(graph, hour=8)
adjusted_graph_2pm = model.apply_time_of_day_costs(graph, hour=14)

print(f"\nGraph edge cost adjustment:")
sample_edge_id = list(graph.edges())[0]
u, v = sample_edge_id
print(f"  Edge {u}-{v}:")
print(f"    - Original cost: {graph[u][v]['cost']:.1f}")
print(f"    - At 8 AM: {adjusted_graph_8am[u][v]['cost']:.1f}")
print(f"    - At 2 PM: {adjusted_graph_2pm[u][v]['cost']:.1f}")

# Option 4: Plan routes at different times and compare
print(f"\nRoute planning with congestion model:")
if graph.number_of_nodes() >= 2:
    node_list = list(graph.nodes())
    start_node = node_list[0]
    end_node = node_list[1]

    print(f"  Route: {start_node} -> {end_node}")
    print(f"    - Static cost: {graph[start_node][end_node].get('cost', 'N/A')}")
    print(f"    - At 8 AM: {adjusted_graph_8am[start_node][end_node]['cost']:.1f}")
    print(f"    - At 2 PM: {adjusted_graph_2pm[start_node][end_node]['cost']:.1f}")
