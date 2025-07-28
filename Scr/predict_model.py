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

# Load model and preprocessing objects
MODEL_PATH = "models/nn_model.keras"
SCALER_PATH = "models/scaler.pkl"
COLUMNS_PATH = "models/model_columns.pkl"

model = load_model(MODEL_PATH, custom_objects={"real_mae": real_mae})
scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(COLUMNS_PATH)

# Field definitions
numeric_fields = [
    "Number of Floors", "Total Area", "Number of Rooms",
    "Number of Bathrooms", "Ceiling Height", "Floor"
]

columns_to_onehot = ["Building Type", "Balcony", "Furniture", "Renovation", "District"]
columns_to_binary = ["New Building", "Elevator"]

# Prediction function
def predict_price(input_dict):
    try:
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical columns (same as train_model.py)
        df_cat = pd.get_dummies(input_df, columns=columns_to_onehot, drop_first=False)
        df_cat = pd.get_dummies(df_cat, columns=columns_to_binary, drop_first=True)

        # Add missing columns
        for col in model_features:
            if col not in df_cat.columns:
                df_cat[col] = 0

        # Ensure column order
        df_cat = df_cat[model_features]

        # Scale numeric fields
        df_cat[numeric_fields] = scaler.transform(df_cat[numeric_fields])

        # Predict
        y_pred_log = model.predict(df_cat)
        predicted_price = np.expm1(y_pred_log[0][0])
        return round(predicted_price, 2)

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

# Example usage
if __name__ == "__main__":
    example_input = {
        "Number of Floors": 10,
        "Total Area": 50,
        "Number of Rooms": 2,
        "Number of Bathrooms": 1,
        "Ceiling Height": 3,
        "Floor": 8,
        "Building Type": "Panel",
        "Balcony": "Closed Balcony",
        "Furniture": "Available",
        "Renovation": "Old Renovation",
        "District": "Kentron",
        "New Building": "No",
        "Elevator": "Available"
    }

    price = predict_price(example_input)
    print(f"Predicted Price: {math.ceil(price)} $")