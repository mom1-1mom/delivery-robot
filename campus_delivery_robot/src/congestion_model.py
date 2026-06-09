"""Traffic congestion model for time-aware route planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

TARGET_COLUMNS = {
    "travel_time",
    "travel_time_s",
    "travel_time_seconds",
    "duration",
    "duration_s",
    "duration_seconds",
}


class TrafficCongestionModel:
    """A lightweight model for predicting edge travel times from historical data."""

    def __init__(self) -> None:
        """Initialize an empty congestion model and its training metadata."""
        self.model: RandomForestRegressor | None = None
        self.feature_columns: list[str] = []
        self.train_rmse: float | None = None
        self.train_samples = 0
        self.trained = False

    def train_from_dataframe(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Train the congestion model from a historical dataset."""
        dataframe = self._prepare_training_data(data)
        features = self._build_feature_matrix(dataframe)
        target = dataframe["travel_time"].astype(float)

        if len(target) < 10:
            raise ValueError("Training data must contain at least 10 samples.")

        self.train_samples = len(target)
        # Keep a validation subset so the UI can report an interpretable RMSE.
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )

        model = RandomForestRegressor(n_estimators=80, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        self.train_rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        self.model = model
        self.feature_columns = list(features.columns)
        self.trained = True

    def _prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize accepted CSV schemas into the model's canonical columns."""
        df = data.copy()
        columns = {column.lower() for column in df.columns}

        # Accept common aliases so user-provided datasets need minimal cleanup.
        if "hour" not in columns:
            for candidate in ("hour", "time_of_day", "departure_hour", "hour_of_day"):
                if candidate in columns:
                    df["hour"] = df[candidate]
                    break
        if "hour" not in df.columns and "timestamp" in columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        if "weekday" not in columns:
            for candidate in ("weekday", "day_of_week", "day"):
                if candidate in columns:
                    df["weekday"] = df[candidate]
                    break
        if "weekday" not in df.columns and "timestamp" in columns:
            df["weekday"] = pd.to_datetime(df["timestamp"]).dt.weekday
        if "weekday" not in df.columns:
            df["weekday"] = 0

        if "highway" not in columns:
            df["highway"] = "unknown"

        if "distance" not in columns:
            raise ValueError("Congestion training data must include a distance column.")

        if "edge_id" not in columns:
            if "way_id" in columns:
                df["edge_id"] = df["way_id"].astype(str)
            elif "u" in columns and "v" in columns:
                df["edge_id"] = df["u"].astype(str) + "_" + df["v"].astype(str)
            else:
                df["edge_id"] = "unknown_edge"

        target_column = next((col for col in df.columns if col.lower() in TARGET_COLUMNS), None)
        if target_column is None:
            raise ValueError(
                "Congestion training data must include a travel time target column such as travel_time or duration."
            )

        df["travel_time"] = pd.to_numeric(df[target_column], errors="coerce")
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
        df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").fillna(0).astype(int)
        df["highway"] = df["highway"].fillna("unknown").astype(str)

        df = df.dropna(subset=["travel_time", "distance"])
        return df

    def _build_feature_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode numeric and categorical fields as model-ready features."""
        features = pd.DataFrame(
            {
                "distance": data["distance"].astype(float),
                "hour": data["hour"].astype(int),
                "weekday": data["weekday"].astype(int),
            }
        )
        highway_dummies = pd.get_dummies(data["highway"].fillna("unknown").astype(str), prefix="highway", dtype=float)
        features = pd.concat([features, highway_dummies], axis=1)

        if "edge_id" in data.columns:
            edge_dummies = pd.get_dummies(data["edge_id"].fillna("unknown_edge").astype(str), prefix="edge", dtype=float)
            features = pd.concat([features, edge_dummies], axis=1)

        return features

    def _edge_feature_record(self, edge_data: dict[str, Any], hour: int, weekday: int) -> dict[str, Any]:
        """Convert one graph edge into the raw feature schema used for inference."""
        return {
            "distance": float(edge_data.get("distance", 0.0)),
            "hour": int(hour),
            "weekday": int(weekday),
            "highway": str(edge_data.get("highway", "unknown") or "unknown"),
            "edge_id": str(edge_data.get("edge_id", edge_data.get("way_id", "unknown_edge")) or "unknown_edge"),
        }

    def _make_edge_features(self, edge_data: dict[str, Any], hour: int, weekday: int) -> pd.DataFrame:
        """Build an aligned feature matrix for one graph edge."""
        return self._make_edges_features([edge_data], hour, weekday)

    def _make_edges_features(self, edge_data_list: list[dict[str, Any]], hour: int, weekday: int) -> pd.DataFrame:
        """Build an aligned feature matrix for a batch of graph edges."""
        sample = pd.DataFrame(
            [self._edge_feature_record(edge_data, hour, weekday) for edge_data in edge_data_list]
        )
        features = self._build_feature_matrix(sample)
        # Match training columns exactly, including one-hot categories not present
        # in the current prediction batch.
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        return features

    def _has_cached_prediction(self, edge_data: dict[str, Any], hour: int, weekday: int) -> bool:
        """Return whether an edge contains a prediction for the requested time."""
        return (
            "predicted_travel_time" in edge_data
            and edge_data.get("predicted_hour") == int(hour)
            and edge_data.get("predicted_weekday") == int(weekday)
        )

    def predict_edge_travel_time(self, edge_data: dict[str, Any], hour: int, weekday: int = 0) -> float:
        """Predict the travel time for a single graph edge."""
        if not self.trained or self.model is None:
            raise ValueError("TrafficCongestionModel has not been trained.")

        if self._has_cached_prediction(edge_data, hour, weekday):
            return float(edge_data["predicted_travel_time"])

        features = self._make_edge_features(edge_data, hour, weekday)
        predicted = self.model.predict(features)[0]
        return float(max(0.1, predicted))

    def predict_edges_travel_time(
        self, edge_data_list: list[dict[str, Any]], hour: int, weekday: int = 0
    ) -> list[float]:
        """Predict travel times for many graph edges in one model call."""
        if not self.trained or self.model is None:
            raise ValueError("TrafficCongestionModel has not been trained.")
        if not edge_data_list:
            return []

        # Batch inference avoids thousands of expensive per-edge predict calls.
        features = self._make_edges_features(edge_data_list, hour, weekday)
        predictions = self.model.predict(features)
        return [float(max(0.1, predicted)) for predicted in predictions]

    def predict_path_travel_time(self, graph: Any, path: list[str], hour: int, weekday: int = 0) -> float:
        """Predict the travel time for a path in the graph."""
        total = 0.0
        uncached_edges: list[dict[str, Any]] = []
        # Reuse graph-level predictions and batch only the remaining edges.
        for u, v in zip(path, path[1:]):
            if graph.has_edge(u, v):
                edge_data = graph[u][v]
                if self._has_cached_prediction(edge_data, hour, weekday):
                    total += float(edge_data["predicted_travel_time"])
                else:
                    uncached_edges.append(edge_data)
        if uncached_edges:
            total += sum(self.predict_edges_travel_time(uncached_edges, hour, weekday))
        return float(total)

    def apply_time_of_day_costs(self, graph: Any, hour: int, weekday: int = 0) -> Any:
        """Return a graph copy with predicted travel-time costs applied to each edge."""
        if not self.trained:
            raise ValueError("TrafficCongestionModel has not been trained.")

        adjusted = graph.copy()
        edge_items = list(adjusted.edges(data=True))
        # Predict the entire graph in one call before assigning route costs.
        predictions = self.predict_edges_travel_time(
            [edge_data for _, _, edge_data in edge_items],
            hour,
            weekday,
        )

        for (_, _, edge_data), predicted_time in zip(edge_items, predictions):
            if "static_cost" not in edge_data:
                edge_data["static_cost"] = float(edge_data.get("cost", edge_data.get("distance", 0.0)))
            edge_data["predicted_travel_time"] = predicted_time
            edge_data["predicted_hour"] = int(hour)
            edge_data["predicted_weekday"] = int(weekday)
            edge_data["cost"] = predicted_time
            edge_data["time_cost"] = predicted_time
        return adjusted

    def report(self) -> dict[str, Any]:
        """Return model status and training metrics for UI presentation."""
        return {
            "trained": self.trained,
            "train_samples": self.train_samples,
            "train_rmse_seconds": self.train_rmse,
            "feature_columns": self.feature_columns,
        }
