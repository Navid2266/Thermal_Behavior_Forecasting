# Thermal Behavior Forecasting

This repository contains a **time-aware machine learning pipeline** for predicting indoor temperatures in residential apartments using sensor data. Built with PyTorch, it handles irregular time series, heating durations, and temporal features. Note: This is a sample of the project; the full project is still in progress and is subject to company confidentiality policies.

---

## Features

### Data Preprocessing
- **Time features**: Extracts hour, day of the week, weekend flag, and cyclic hour transformations (`hour_sin`, `hour_cos`) using `add_time_features()`.
- **Heating duration**: Computes continuous heating durations with `add_heating_duration()`, resetting at gaps in the time series.

### Dataset Construction
- `TimeAwareSeriesDataset` (PyTorch Dataset) prepares:
  - Input sequences (history windows) and target outputs (future temperature).
  - Handles missing data and gaps exceeding `max_gap_minutes`.
  - Normalizes continuous features with `MinMaxScaler`.

### Model
- `TimeSeriesMLP` (Multilayer Perceptron for time series):
  - Compresses temporal sequence with a linear layer before flattening.
  - Fully connected layers predict future indoor temperature.

### Training
- Splits data into **train/test sets (80/20)**.
- Supports **GPU training** with Automatic Mixed Precision (AMP).
- Early stopping is applied based on validation loss.
- Saves both **best model** (`best_model.pth`) and **final model** (`final_best_model.pth`).

### Evaluation & Visualization
- `collect_predictions()` runs the trained model on new data and optionally inverse-scales outputs.
- Results can be visualized to compare **predicted vs actual temperatures**.

---

## 📈 Example Performance

Unseen window predictions (Feb 27 – Mar 3, 2025):

![Prediction vs Actual](https://github.com/Navid2266/Thermal_Behavior_Forecasting/blob/main/plots/Screenshot%202025-10-18%20230113.png)

The plot shows that the **predicted temperatures closely follow the actual indoor temperature**, demonstrating the model’s accuracy in capturing temporal patterns and heating dynamics.

---

## Usage
1. Place your CSV sensor data in the project folder.
2. Update the data path in `load_data()` function.
3. Run preprocessing functions and create datasets.
4. Initialize `TimeSeriesMLP` and train using the provided training loop.
5. Evaluate and visualize predictions.

---

## Workflow

Raw Sensor Data → add_time_features → add_heating_duration → TimeAwareSeriesDataset → TimeSeriesMLP → Predictions → Visualization

---

## Notes
- Modular design: easy to adapt for different apartments or datasets.
- Handles irregular time intervals and missing data gracefully.
- Supports configurable history windows and prediction horizons.
