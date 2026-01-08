import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

import joblib

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
data = pd.read_csv("processed_combined_accidents.csv")

# -------------------------------------------------
# 2. FEATURE ENGINEERING (CYCLIC TIME FEATURES)
# -------------------------------------------------
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

data["day_sin"] = np.sin(2 * np.pi * data["day"] / 7)
data["day_cos"] = np.cos(2 * np.pi * data["day"] / 7)

# -------------------------------------------------
# 3. SELECT FEATURES & TARGET
# -------------------------------------------------
X = data[
    [
        "latitude",
        "longitude",
        "traffic_volume",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos"
    ]
]

y = data["accident"]

# -------------------------------------------------
# 4. TRAIN / VALIDATION / TEST SPLIT (80 / 10 / 10)
# -------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Dataset split:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# -------------------------------------------------
# 5. FEATURE SCALING (FIT ONLY ON TRAIN)
# -------------------------------------------------
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 6. RESHAPE FOR LSTM
# -------------------------------------------------
X_train_scaled = X_train_scaled.reshape(
    (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
)
X_val_scaled = X_val_scaled.reshape(
    (X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
)
X_test_scaled = X_test_scaled.reshape(
    (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
)

# -------------------------------------------------
# 7. HANDLE CLASS IMBALANCE
# -------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# -------------------------------------------------
# 8. BUILD LSTM MODEL
# -------------------------------------------------
model = Sequential([
    Input(shape=(1, X_train_scaled.shape[2])),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------
# 9. TRAIN WITH EARLY STOPPING
# -------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# -------------------------------------------------
# 10. TRAINING CURVES
# -------------------------------------------------
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# -------------------------------------------------
# 11. FINAL EVALUATION ON TEST SET
# -------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
# -------------------------------------------------
# 12. SAVE MODEL & SCALER
# -------------------------------------------------
model.save("traffic_lstm.keras")
joblib.dump(scaler, "scaler.pkl")

print("\nModel trained, evaluated, and saved successfully")
