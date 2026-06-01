## Traffic Congestion Model Integration

### Overview

This project now includes a **machine learning-based traffic congestion model** that predicts edge travel times based on historical congestion data. The model enables **time-aware route planning** - generating different optimal routes for different times of day.

---

## How It Works

### 1. **Model Architecture**

The `TrafficCongestionModel` class uses **Random Forest Regression** to predict travel times:

- **Input Features**:
  - `distance` - Edge length in meters
  - `hour` - Hour of day (0-23)
  - `weekday` - Day of week (0-6)
  - `highway` - Road type (footway, residential, service, etc.)

- **Target**: `travel_time` - Predicted travel duration in seconds

- **Training**: 80% training / 20% validation split with RMSE evaluation

### 2. **Integration with Route Planning**

**Without congestion model**:
- Routes optimized using static edge costs (based on distance and road type)
- Same route for any time of day

**With congestion model**:
- Upload historical congestion CSV
- Model trains on data and learns time-dependent cost patterns
- Routes adjusted based on predicted travel times at departure hour
- Different hours yield different optimal routes

---

## Usage Guide

### Step 1: Prepare Your Historical Data

Create a CSV file with the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `distance` | float | ✓ | Edge distance in meters |
| `travel_time` | float | ✓ | Observed travel time (seconds) |
| `hour` | int | ✓ | Hour of day (0-23) |
| `weekday` | int | ✓ | Day of week (0=Monday, 6=Sunday) |
| `highway` | string | ✓ | Road type (footway, residential, service, path, living_street, pedestrian, unclassified) |
| `way_id`, `u`, `v` | - | ✗ | Optional identifiers |

**Example CSV Structure**:
```csv
way_id,u,v,distance,highway,hour,weekday,travel_time
1,node_A,node_B,120,footway,8,2,45
1,node_A,node_B,120,footway,9,2,50
1,node_A,node_B,120,footway,14,2,58
2,node_B,node_C,85,residential,8,2,38
...
```

**Sample data**: See `campus_delivery_robot/data/sample_congestion.csv`

### Step 2: Upload and Train

1. Open the Streamlit app
2. Expand **"Traffic congestion model"** in the sidebar
3. Click **"Upload historical congestion CSV"**
4. Select your CSV file - the model trains automatically
5. Set **"Departure hour"** (0-23)
6. You'll see training status: `Trained on X samples | Validation RMSE: Y sec`

### Step 3: Generate Routes

1. Select start location and delivery points normally
2. Click **"Generate Multi-stop Route"**
3. The planner now uses **predicted travel times** for your selected hour
4. Results show:
   - **Estimated Delivery Time**: Based on trained model predictions
   - **Comparison**: Baseline route vs. ML-optimized route (if both feasible)

---

## Output Interpretation

### Metrics

- **Estimated Delivery Time**: Total travel duration in minutes (using predicted times)
- **Total Distance**: Sum of edge distances (unchanged by model)
- **Predicted Fee**: Calculated from distance and stop count
- **Training RMSE**: Model validation error in seconds
  - Lower RMSE = better predictions
  - ~5-10 sec is typical for campus-scale data

### Route Comparison

When a congestion model is active:

- **Baseline Route**: Optimized using static costs (distance-based)
- **ML-Optimized Route**: Optimized using predicted travel times for the selected hour
- If delivery order differs → model found a faster route for that time

---

## Technical Details

### Model Parameters

- **Algorithm**: Random Forest (80 trees)
- **Feature Engineering**: One-hot encoding for road types
- **Train/Test Split**: 80/20 stratified
- **Hyperparameters**: Fixed (can be extended)

### Cost Application

Each graph edge gets:

```python
edge["static_cost"] = original distance-based cost
edge["predicted_travel_time"] = ML prediction for selected hour
edge["cost"] = predicted_travel_time  # Used by routing algorithm
```

The routing algorithm (BFS/UCS/A*) then optimizes using predicted times.

---

## Example Workflow

1. Collect GPS traces from campus robots over several weeks
2. For each road segment + time slot, record:
   - Distance
   - Observed travel time
   - Hour and weekday
   - Road type (from OSM data)
3. Save as `congestion.csv`
4. In Streamlit:
   - Upload `congestion.csv`
   - Set departure hour to `14` (2 PM)
   - Generate route
5. Observe optimizations:
   - Rush hours → routes avoid congested segments
   - Off-peak hours → shorter paths prioritized
   - Road types considered → faster routes on footways, slower on service roads

---

## Files

- **Model Class**: `campus_delivery_robot/src/congestion_model.py`
- **App Integration**: `campus_delivery_robot/app.py` (sidebar + route planning)
- **Sample Data**: `campus_delivery_robot/data/sample_congestion.csv`

---

## Future Extensions

- **Time Window Optimization**: Different costs for delivery time windows
- **Weather Data**: Include weather features for better predictions
- **Real-time Updates**: Retrain model with fresh GPS data
- **Hyperparameter Tuning**: Grid search over model parameters
- **Ensemble Methods**: Combine multiple models for robustness
- **Confidence Intervals**: Report uncertainty in predictions

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Training data must contain at least 10 samples" | Increase CSV rows (need ≥10 samples) |
| "travel_time column not found" | Ensure CSV has column named `travel_time`, `duration`, or `duration_s` |
| "distance column required" | Add `distance` column (in meters) |
| Model trains but route unchanged | Model may have low feature importance; check data quality |

