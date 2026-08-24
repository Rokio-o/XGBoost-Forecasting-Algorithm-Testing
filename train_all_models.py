"""
Train and Evaluate Forecasting Models

This script loads the processed CO2 forecaseting features, separates the input features from the one-hour-ahead prediction target,
and divides the data chronologically into training and testing sets. 
It then trains and evaluates seven forecasting algorithms which are: ARIMA, Random Forest, SVR, XGBoost, LightBGM, MLP, and LSTM.
The evaluation will be based on the MSE, RMSE, MAPE, and R2 metrics (based on GeeksforGeeks ML Model Evaluation Guide)
The results are compiled into a comparison table to determine the perforamance of each model.

"""
import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import numpy as np # for numerical operations
import pandas as pd # loading and manipulating tabular data

from sklearn.ensemble import RandomForestRegressor # for Random Forest
from sklearn.svm import SVR # for Support Vector Regression
from sklearn.neural_network import MLPRegressor # for Multi-Layer Perceptron
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_absolute_percentage_error, r2_score) # for model evaluation metrics 

from xgboost import XGBRegressor # for XGBoost

from lightgbm import LGBMRegressor # for LightGBM

from statsmodels.tsa.arima.model import ARIMA # for ARIMA

import tensorflow as tf
from tensorflow import keras

# will collect each model's scores and store in this dictionary for comparison
RESULTS = {}

def evaluate(name, y_true, y_pred):
    """
    Computes the 5 standard regression metrics for one model's predictions
    and both prints them AND sotres them in the RESULTS for the final table.

    y_true = REAL values that happend (ground truth)
    y_pred = PREDICTED values from the model (guessed values)
    """

    mae = mean_absolute_error(y_true, y_pred) # average of (real - predicted) absolute values
    mse  = mean_squared_error(y_true, y_pred) # average of (real - predicted)^2 values
    rmse = np.sqrt(mse) # square root of MSE
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100 # average of (real - predicted)/real absolute values
    r2 = r2_score(y_true, y_pred) # R-squared value (1 is perfect, 0 is no better than mean)

    # store the results in the RESULTS dictionary for final comparison table
    RESULTS[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

    # print formatted results for this model
    print(f"{name:15s} | MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%  R2={r2:.3f}")

# ==================================================================================================================
# LOAD FEATURES (built earlier in feature_engineering.py)
# ==================================================================================================================

# read CSV file that was created in feature_engineering.py
data = pd.read_csv("features_co2_N01.csv", parse_dates=["timestamp"], index_col="timestamp")

# FEATURE_COLS = every column except the target and the next-hour prediction (y)
# 'y' is what to predict, it should never appear in the model's input
FEATURE_COLS = [c for c in data.columns if c not in ["y"]]

X = data[FEATURE_COLS] # input features
y = data["y"] # next-hour prediction target

# ==================================================================================================================
# TRAIN/TEST SPLIT - chronological split (no randomization), NEVER shuffle time series data
# ==================================================================================================================

# we will use the first 85% of the data for training and the last 15% for testing
split_idx = int(len(data) * 0.85)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:] # train/test split for input features
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:] # train/test split for prediction target

# sanity check to make sure the split is correct, confirm split sizes and dates looks right
print(f"Train size: {len(X_train)} Test size: {len(X_test)}")
print(f"Test period: {X_test.index.min()} to {X_test.index.max()}\n")


# ----------------------------------------------------------------------
# 1) ARIMA  (univariate - only uses past CO2 values, no other features)
# ----------------------------------------------------------------------
train_series = data["target"].iloc[:split_idx]
test_series = data["target"].iloc[split_idx:]  # we forecast t+1 = the 'y' column, but ARIMA needs history

# Fit on the raw target history, then forecast one step at a time (walk-forward)
history = list(train_series.values)
arima_preds = []
for t in range(len(y_test)):
    model = ARIMA(history, order=(2, 1, 2))
    fitted = model.fit()
    yhat = fitted.forecast(steps=1)[0]
    arima_preds.append(yhat)
    history.append(test_series.values[t])  # feed true value forward (walk-forward validation)
evaluate("ARIMA", y_test.values, np.array(arima_preds))

# ----------------------------------------------------------------------
# 2) Random Forest
# ----------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test.values, rf.predict(X_test))

# ----------------------------------------------------------------------
# 3) SVR (needs feature scaling!)
# ----------------------------------------------------------------------
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
X_train_s = scaler_X.transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train.values.reshape(-1, 1)).ravel()

svr = SVR(kernel="rbf", C=10, epsilon=0.1)
svr.fit(X_train_s, y_train_s)
svr_pred_scaled = svr.predict(X_test_s)
svr_pred = scaler_y.inverse_transform(svr_pred_scaled.reshape(-1, 1)).ravel()
evaluate("SVR", y_test.values, svr_pred)

# ----------------------------------------------------------------------
# 4) XGBoost
# ----------------------------------------------------------------------
xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
evaluate("XGBoost", y_test.values, xgb.predict(X_test))

# ----------------------------------------------------------------------
# 5) LightGBM
# ----------------------------------------------------------------------
lgbm = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
lgbm.fit(X_train, y_train)
evaluate("LightGBM", y_test.values, lgbm.predict(X_test))

# ----------------------------------------------------------------------
# 6) MLP (Multi-Layer Perceptron - simple neural network)
# ----------------------------------------------------------------------
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                    max_iter=2000, random_state=42, early_stopping=True)
mlp.fit(X_train_s, y_train_s)
mlp_pred = scaler_y.inverse_transform(mlp.predict(X_test_s).reshape(-1, 1)).ravel()
evaluate("MLP", y_test.values, mlp_pred)

# ----------------------------------------------------------------------
# 7) LSTM (needs 3D sequences: [samples, timesteps, features])
# ----------------------------------------------------------------------
SEQ_LEN = 24  # use past 24 hours to predict next hour

def make_sequences(X_arr, y_arr, seq_len):
    Xs, ys = [], []
    for i in range(len(X_arr) - seq_len):
        Xs.append(X_arr[i:i + seq_len])
        ys.append(y_arr[i + seq_len])
    return np.array(Xs), np.array(ys)

X_all_s = scaler_X.transform(X)  # scale full feature set consistently
y_all_s = scaler_y.transform(y.values.reshape(-1, 1)).ravel()

X_seq, y_seq = make_sequences(X_all_s, y_all_s, SEQ_LEN)
split_seq = split_idx - SEQ_LEN
X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]

tf.random.set_seed(42)
lstm = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN, X_train_seq.shape[2])),
    keras.layers.LSTM(32, activation="tanh"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32, verbose=0,
         validation_split=0.1,
         callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

lstm_pred_scaled = lstm.predict(X_test_seq, verbose=0).ravel()
lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
y_test_seq_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
evaluate("LSTM", y_test_seq_actual, lstm_pred)

# ----------------------------------------------------------------------
# FINAL COMPARISON TABLE
# ----------------------------------------------------------------------
print("\n=== FINAL MODEL COMPARISON (target: CO2, 1-hour-ahead) ===")
results_df = pd.DataFrame(RESULTS).T
results_df = results_df.sort_values("RMSE")
print(results_df.round(3))
results_df.to_csv("model_comparison_results.csv")