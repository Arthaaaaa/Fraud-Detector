import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neural_network import MLPClassifier
import joblib

# =========================
# 1) Load dataset
# =========================
df = pd.read_csv("creditcard.csv")

# =========================
# 2) Pilih 3 fitur (sesuai TP-11: 3 input)
#    - Time  -> Waktu transaksi
#    - Amount -> Nominal transaksi
#    - V14 -> "pola transaksi" (dataset ini anonim, jadi kita anggap fitur perilaku)
# =========================
FEATURES = ["Time", "Amount", "V14"]
TARGET = "Class"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# =========================
# 3) Split data: train & test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # penting karena data fraud imbalanced
)

# =========================
# 4) Scaling (WAJIB untuk ANN)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5) Model ANN sesuai desain TP-11:
#    1 hidden layer, 5 neuron, ReLU
#    Output sigmoid (di sklearn pakai logistic untuk biner)
# =========================
model = MLPClassifier(
    hidden_layer_sizes=(5,),   # 1 hidden layer isi 5 neuron
    activation="relu",
    solver="adam",
    max_iter=50,               # boleh 50-100 sesuai modul
    random_state=42
)

# Train
model.fit(X_train_scaled, y_train)

# =========================
# 6) Evaluasi
# =========================
y_pred = model.predict(X_test_scaled)

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=4))

# =========================
# 7) Simpan model & scaler
# =========================
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\nSelesai! model.joblib dan scaler.joblib sudah dibuat.")
