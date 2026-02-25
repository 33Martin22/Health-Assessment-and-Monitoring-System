"""
train_model.py - Reference training script

The trained model (risk_model.h5) and scaler (scaler.pkl) are already
provided and placed in app/models/.  This script documents exactly how they
were produced so results can be reproduced or re-trained on new data.

Usage (if you want to retrain):
    pip install tensorflow scikit-learn pandas numpy
    python train_model.py
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(__file__), "Health_Risk_Dataset_1.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "app", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")
print(f"Risk distribution:\n{df['Risk_Level'].value_counts()}\n")

# ── Target: map Normal → Low so we keep 3 classes ─────────────────────────────
df["Risk_Level"] = df["Risk_Level"].replace({"Normal": "Low"})
print(f"After merging Normal→Low:\n{df['Risk_Level'].value_counts()}\n")

# ── Feature engineering ───────────────────────────────────────────────────────
NUMERIC_COLS = ["Respiratory_Rate", "Oxygen_Saturation", "O2_Scale",
                "Systolic_BP", "Heart_Rate", "Temperature"]

# Scale numeric vitals
scaler = MinMaxScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

# One-hot encode Consciousness (drop_first=True → base = 'A')
ohe = pd.get_dummies(df["Consciousness"], prefix="consciousness", drop_first=True)
df  = pd.concat([df, ohe], axis=1)

FEATURE_COLS = NUMERIC_COLS + list(ohe.columns) + ["On_Oxygen"]
X = df[FEATURE_COLS].values.astype(np.float32)

# Label encode target (alphabetical → High=0, Low=1, Medium=2)
le = LabelEncoder()
y  = le.fit_transform(df["Risk_Level"])
print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Feature vector size: {X.shape[1]}\n")

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Build Keras model ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,  activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32,  activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16,  activation="relu"),
    tf.keras.layers.Dense(3,   activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save artifacts ────────────────────────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, "risk_model.h5"))
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✅ risk_model.h5 saved to {MODEL_DIR}")
print(f"✅ scaler.pkl    saved to {MODEL_DIR}")
print("\nRun the app: streamlit run app/main.py")
