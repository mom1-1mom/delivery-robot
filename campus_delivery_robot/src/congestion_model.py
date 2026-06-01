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
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )

        model = RandomForestRegressor(n_estimators=80, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        self.train_rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        self.model = model
        self.feature_columns = list(features.columns)
        self.trained = True

    def _prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        columns = {column.lower() for column in df.columns}

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

    def _make_edge_features(self, edge_data: dict[str, Any], hour: int, weekday: int) -> pd.DataFrame:
        sample = pd.DataFrame(
            [
                {
                    "distance": float(edge_data.get("distance", 0.0)),
                    "hour": int(hour),
                    "weekday": int(weekday),
                    "highway": str(edge_data.get("highway", "unknown") or "unknown"),
                    "edge_id": str(edge_data.get("edge_id", edge_data.get("way_id", "unknown_edge")) or "unknown_edge"),
                }
            ]
        )
        features = self._build_feature_matrix(sample)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        return features

    def predict_edge_travel_time(self, edge_data: dict[str, Any], hour: int, weekday: int = 0) -> float:
        """Predict the travel time for a single graph edge."""
        if not self.trained or self.model is None:
            raise ValueError("TrafficCongestionModel has not been trained.")

        features = self._make_edge_features(edge_data, hour, weekday)
        predicted = self.model.predict(features)[0]
        return float(max(0.1, predicted))

    def predict_path_travel_time(self, graph: Any, path: list[str], hour: int, weekday: int = 0) -> float:
        """Predict the travel time for a path in the graph."""
        total = 0.0
        for u, v in zip(path, path[1:]):
            if graph.has_edge(u, v):
                total += self.predict_edge_travel_time(graph[u][v], hour, weekday)
        return float(total)

    def apply_time_of_day_costs(self, graph: Any, hour: int, weekday: int = 0) -> Any:
        """Return a graph copy with predicted travel-time costs applied to each edge."""
        if not self.trained:
            raise ValueError("TrafficCongestionModel has not been trained.")

        adjusted = graph.copy()
        for _, _, edge_data in adjusted.edges(data=True):
            if "static_cost" not in edge_data:
                edge_data["static_cost"] = float(edge_data.get("cost", edge_data.get("distance", 0.0)))
            predicted_time = self.predict_edge_travel_time(edge_data, hour, weekday)
            edge_data["predicted_travel_time"] = predicted_time
            edge_data["cost"] = predicted_time
            edge_data["time_cost"] = predicted_time
        return adjusted

    def report(self) -> dict[str, Any]:
        return {
            "trained": self.trained,
            "train_samples": self.train_samples,
            "train_rmse_seconds": self.train_rmse,
            "feature_columns": self.feature_columns,
        }
