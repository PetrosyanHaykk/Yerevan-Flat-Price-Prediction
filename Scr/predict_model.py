# Suppress TensorFlow Logs
import os
import sys
import contextlib
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to silence all stderr output (used for TensorFlow)."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import math
import tensorflow as tf
from keras.models import load_model


# Custom metric
def real_mae(y_true, y_pred):
    y_true_exp = tf.math.expm1(y_true)
    y_pred_exp = tf.math.expm1(y_pred)
    return tf.reduce_mean(tf.abs(y_true_exp - y_pred_exp))


# Paths
current_path = os.path.abspath(__file__)
project_root = current_path[:current_path.find("Yerevan-Flat-Price-Prediction") + len("Yerevan-Flat-Price-Prediction")]

MODEL_PATH = os.path.join(project_root, "models", "nn_model.keras")
SCALER_PATH = os.path.join(project_root, "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(project_root, "models", "model_columns.pkl")


# Load trained model and preprocessing tools
with suppress_stderr():
    model = load_model(MODEL_PATH, custom_objects={"real_mae": real_mae})

scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(COLUMNS_PATH)


# Fields
numeric_fields = [
    "Number of Floors", "Total Area", "Number of Rooms",
    "Number of Bathrooms", "Ceiling Height", "Floor"
]

columns_to_onehot = ["Building Type", "Balcony", "Furniture", "Renovation", "District"]
columns_to_binary = ["New Building", "Elevator"]


# Prediction function
def predict_price(input_dict):
    try:
        input_df = pd.DataFrame([input_dict])
        df_cat = pd.get_dummies(input_df, columns=columns_to_onehot, drop_first=False)
        df_cat = pd.get_dummies(df_cat, columns=columns_to_binary, drop_first=True)

        for col in model_features:
            if col not in df_cat.columns:
                df_cat[col] = 0

        df_cat = df_cat[model_features]
        df_cat[numeric_fields] = scaler.transform(df_cat[numeric_fields])

        with suppress_stderr():
            y_pred_log = model.predict(df_cat)

        predicted_price = np.expm1(y_pred_log[0][0])
        return round(predicted_price, 2)

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")


# Example usage
if __name__ == "__main__":
    example_input = {
        "Number of Floors": 14,
        "Total Area": 60,
        "Number of Rooms": 3,
        "Number of Bathrooms": 2,
        "Ceiling Height": 3,
        "Floor": 8,
        "Building Type": "Monolith",
        "Balcony": "Closed Balcony",
        "Furniture": "Available",
        "Renovation": "Designer Renovated",
        "District": "Kentron",
        "New Building": "Yes",
        "Elevator": "Available"
    }

    price = predict_price(example_input)
    print(f"Predicted Price: {math.ceil(price)} $")