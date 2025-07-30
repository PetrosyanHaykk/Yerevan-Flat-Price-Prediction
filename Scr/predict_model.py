# Import necessary libraries
import pandas as pd
import numpy as np
import os
import joblib
import math
import tensorflow as tf
from keras.models import load_model

# Custom metric used during training (must be redefined for loading the model)
def real_mae(y_true, y_pred):
    y_true_exp = tf.math.expm1(y_true)
    y_pred_exp = tf.math.expm1(y_pred)
    return tf.reduce_mean(tf.abs(y_true_exp - y_pred_exp))

# Paths to the saved model and preprocessing objects

current_path = os.getcwd()
print("CURRENT PATH:", current_path)

MODEL_PATH = os.path.join(current_path, "models", "nn_model.keras")
SCALER_PATH = os.path.join(current_path, "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(current_path, "models", "model_columns.pkl")

# Load trained model and preprocessing tools
model = load_model(MODEL_PATH, custom_objects={"real_mae": real_mae})
scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(COLUMNS_PATH)

# Numeric fields that need scaling
numeric_fields = [
    "Number of Floors", "Total Area", "Number of Rooms",
    "Number of Bathrooms", "Ceiling Height", "Floor"
]

# Categorical fields to one-hot encode
columns_to_onehot = ["Building Type", "Balcony", "Furniture", "Renovation", "District"]

# Binary fields (converted to 0/1)
columns_to_binary = ["New Building", "Elevator"]

# Function to predict apartment price
def predict_price(input_dict):
    try:
        # Convert input dictionary into a DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical and binary columns (same logic as training)
        df_cat = pd.get_dummies(input_df, columns=columns_to_onehot, drop_first=False)
        df_cat = pd.get_dummies(df_cat, columns=columns_to_binary, drop_first=True)

        # Add any missing columns that were present during training
        for col in model_features:
            if col not in df_cat.columns:
                df_cat[col] = 0

        # Ensure the column order matches the training data
        df_cat = df_cat[model_features]

        # Scale the numeric fields using the saved scaler
        df_cat[numeric_fields] = scaler.transform(df_cat[numeric_fields])

        # Make a prediction (log-transformed), then invert the log transform
        y_pred_log = model.predict(df_cat)
        predicted_price = np.expm1(y_pred_log[0][0])

        # Return the price rounded to 2 decimals
        return round(predicted_price, 2)

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

# Example usage of the prediction function
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