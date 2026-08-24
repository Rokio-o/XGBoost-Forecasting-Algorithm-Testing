"""
Feature Engineeing for 1-Hour-Ahead CO2 Forecasting

This script loads the Aerintel synthetic sensor dataset then selects the N01 sensor node data (classroom).
After selecting, it organize the data chronologically and then prepares the features needed for forecasting.
It then creates time-based, lag, and rolling features from historical sensor readings (synthetic),
includes supporitng sensor variables such as temperature, humidity, and occupancy count, and defines the next CO2
value as the prediction target.
The processed dataset is cleaned and saved for use in training and evaluating the foreacasting algorithms/models.
"""

import numpy as np # for numerical operations
import pandas as pd # loading and manipulating tabular data


# load the dataset 
df = pd.read_csv("aerintel_synthetic_dataset-forecasting.csv", parse_dates=["timestamp"])

# load sensor node records for node id = N01
node = df[df.node_id == "N01"]. copy()

# put the observations in chronoligical order then reset the row numbers after sorting
node = node.sort_values("timestamp").reset_index(drop=True)

# will make the timestamp column the index of the dataframe
node = node.set_index("timestamp")

# target pollutant for forecasting (PM2.5 / CO2), chage the target either of the two
TARGET = "co2"

# function to extract information from the timestamp and create new features
def build_features(data, target_col):
    d = data.copy()
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)

    # lag features for historical information
    for lag in [1, 2, 3, 24]:
        d[f"{target_col}_lag{lag}"] = d[target_col].shift(lag)

   # rolling window features (past 3h, 6h averages - shifted so no leakage)
    d[f"{target_col}_roll3"] = d[target_col].shift(1).rolling(3).mean()
    d[f"{target_col}_roll6"] = d[target_col].shift(1).rolling(6).mean()

    # supporting sensor features (same timestamp is OK, these are inputs not the target)
    d_feat = d[["hour", "dayofweek", "is_weekend",
               f"{target_col}_lag1", f"{target_col}_lag2", f"{target_col}_lag3", f"{target_col}_lag24",
               f"{target_col}_roll3", f"{target_col}_roll6",
               "temperature_c", "humidity_pct", "occupancy_count"]].copy()

    d_feat["target"] = d[target_col] # value at time t (used to build t+1 label below)
    d_feat["y"] = d[target_col].shift(-1) # wanted prediction which is the next hour (hourly prediction)

    d_feat = d_feat.dropna() # drop rows with missing values
    return d_feat

data = build_features(node, TARGET)
print("Shape after feature engineering:", data.shape)
print(data.head())
data.to_csv("features_co2_N01.csv")





