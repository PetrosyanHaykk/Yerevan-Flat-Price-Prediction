# Import necessary libraries
import os
import random
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input, PReLU
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping, Callback

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set seeds for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# Load dataset
current_path = os.getcwd()
print("CURRENT PATH:", current_path)

csv_path = os.path.join(current_path, "Data", "processed", "processed_df.csv")
df = pd.read_csv(csv_path)
print(f"Data loaded from: {csv_path}")

# One-Hot Encoding of categorical columns
columns_to_onehot = ["Building Type", "Balcony", "Furniture", "Renovation", "District"]
columns_to_binary = ["New Building", "Elevator"]

df = pd.get_dummies(df, columns=columns_to_onehot, drop_first=False)
df = pd.get_dummies(df, columns=columns_to_binary, drop_first=True)

# Shuffle and split
df_shuffled = shuffle(df, random_state=42).reset_index(drop=True)
X = df_shuffled.drop("Price", axis=1)
y = df_shuffled["Price"].values
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, random_state=42)

# Scale numeric features
numeric_columns = ['Number of Floors', 'Total Area', 'Number of Rooms',
                   'Number of Bathrooms', 'Ceiling Height', 'Floor']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Custom metric
def real_mae(y_true, y_pred):
    y_true_exp = tf.math.expm1(y_true)
    y_pred_exp = tf.math.expm1(y_pred)
    return tf.reduce_mean(tf.abs(y_true_exp - y_pred_exp))

# Learning Rate Scheduler
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=15,
    t_mul=2.0,
    m_mul=0.9,
    alpha=1e-6
)

# Build Model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(518, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    PReLU(),
    Dropout(0.4),
    
    Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    PReLU(),
    Dropout(0.3),
    
    Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    PReLU(),
    Dropout(0.3),
    
    Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    PReLU(),
    Dropout(0.1),
    
    Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    PReLU(),
    
    Dense(1)
])

model.compile(
    optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
    loss='mean_squared_error',
    metrics=[real_mae]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=1e-3)

class LRTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"Epoch {epoch+1}: Learning Rate = {lr:.6f}")

# Train Model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, LRTracker()],
    verbose=1
)

# Evaluate Model
y_pred_log = model.predict(X_test_scaled).flatten()
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
r2 = r2_score(y_test_real, y_pred)

print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R² Score: {r2:.3f}")

# Create output folder
os.makedirs("models", exist_ok=True)

# Plot MAE During Training and save
plt.figure(figsize=(8, 5))
plt.plot(history.history['real_mae'], label='Train MAE (Real Prices)', linewidth=2)
plt.plot(history.history['val_real_mae'], label='Val MAE (Real Prices)', linewidth=2)
plt.title('MAE During Training', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('MAE (Real Prices)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("models/mae_training.png", dpi=300)
plt.show()

# Plot Real vs Predicted Prices and save
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_pred, color="purple", alpha=0.6, edgecolor="k")
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()],
         color="red", linestyle="--", linewidth=2)
plt.title("Real vs Predicted Prices", fontsize=16, fontweight='bold')
plt.xlabel("Real Price (USD)", fontsize=12)
plt.ylabel("Predicted Price (USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("models/real_vs_predicted.png", dpi=300)
plt.show()

# Plot Residuals Distribution and save
residuals = y_test_real - y_pred

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=60, kde=True, color='purple', alpha=0.3)
plt.title("Residuals Distribution (Actual Price - Predicted Price)", fontsize=15, fontweight='bold')
plt.xlabel("Residual (USD)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("models/residuals_distribution.png", dpi=300)
plt.show()

# Save Model and Preprocessing Artifacts
model.save("models/nn_model.keras", include_optimizer=False)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(df_shuffled.drop("Price", axis=1).columns), "models/model_columns.pkl")

print("Model, scaler, feature columns, and plots saved in 'models/' folder.")