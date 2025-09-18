# ====================================================================
# Prediction_Of_Sepsis_GA_Optimized_GRU.py
# Train & Evaluate GRU Model using Best Hyperparameters from GA
# ====================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report, confusion_matrix
)

# ====================================================================
# Step 1: Load Best Hyperparameters from GA_Optimize_Sepsis
# ====================================================================
GA_PARAMS_FILE = "GA_Optimize_Sepsis_BestHyperparams.json"

with open(GA_PARAMS_FILE, "r") as f:
    params = json.load(f)

print("[INFO] Using best hyperparameters from GA optimization:")
print(json.dumps(params, indent=2))

# ====================================================================
# Step 2: Dataset Preparation
# ====================================================================
CSV_PATH = "Dataset_Processed.csv"
df = pd.read_csv(CSV_PATH)

features = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
target = "SepsisLabel"

# Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
SCALER_FILE = "Prediction_Of_Sepsis_GA_Optimized_GRU_Scaler.pkl"
joblib.dump(scaler, SCALER_FILE)

# Group by patient
patients = []
for pid, group in df.groupby("Patient_ID"):
    X = group[features].values
    y = group[target].values
    patients.append((X, y))

# Split into Train / Validation / Test
train_patients, temp_patients = train_test_split(patients, test_size=0.3, random_state=42)
val_patients, test_patients   = train_test_split(temp_patients, test_size=0.5, random_state=42)

print(f"[INFO] Patients split -> Train: {len(train_patients)}, "
      f"Val: {len(val_patients)}, Test: {len(test_patients)}")

# ====================================================================
# Step 3: Dataset Class
# ====================================================================
class SepsisDataset(Dataset):
    """Custom Dataset for sequential patient data."""
    def __init__(self, patient_data, seq_len=12):
        self.data = []
        for X, y in patient_data:
            if len(X) >= seq_len:
                for i in range(len(X)-seq_len+1):
                    self.data.append((X[i:i+seq_len], y[i+seq_len-1]))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ====================================================================
# Step 4: GRU Model
# ====================================================================
class GRUNet(nn.Module):
    """GRU-based model for sepsis prediction."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True,
                          dropout=(dropout if num_layers > 1 else 0))
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze()

# ====================================================================
# Step 5: DataLoaders
# ====================================================================
train_loader = DataLoader(SepsisDataset(train_patients), batch_size=64, shuffle=True)
val_loader   = DataLoader(SepsisDataset(val_patients), batch_size=64)
test_loader  = DataLoader(SepsisDataset(test_patients), batch_size=64)

# ====================================================================
# Step 6: Model, Loss, Optimizer
# ====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GRUNet(
    input_dim=len(features),
    hidden_dim=params["hidden_dim"],
    num_layers=params["num_layers"],
    dropout=params["dropout"]
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

# ====================================================================
# Step 7: Training with Early Stopping
# ====================================================================
MODEL_FILE = "Prediction_Of_Sepsis_GA_Optimized_GRU_BestModel.pt"

best_auc, patience, patience_counter = 0, 5, 0
EPOCHS = 20

print("[INFO] Starting training...")

for epoch in range(EPOCHS):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            probs = torch.sigmoid(model(X_batch.to(device)))
            y_true.extend(y_batch.numpy())
            y_prob.extend(probs.cpu().numpy())
    auc = roc_auc_score(y_true, y_prob)
    print(f"[Epoch {epoch+1:02d}] Val AUC={auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), MODEL_FILE)
        patience_counter = 0
        print(f"[INFO] New best model saved (AUC={auc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("[INFO] Early stopping triggered")
            break

# ====================================================================
# Step 8: Final Test Evaluation
# ====================================================================
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        probs = torch.sigmoid(model(X_batch.to(device)))
        y_true.extend(y_batch.numpy())
        y_prob.extend(probs.cpu().numpy())

# Best threshold from PR curve
prec, rec, thr = precision_recall_curve(y_true, y_prob)
f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
best_thr = thr[np.argmax(f1_scores)]
y_pred = (np.array(y_prob) >= best_thr).astype(int)

# Metrics
results = {
    "Best_Threshold": float(best_thr),
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "ROC AUC": roc_auc_score(y_true, y_prob),
    "PR AUC": average_precision_score(y_true, y_prob),
    "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
    "Classification Report": classification_report(y_true, y_pred, output_dict=True),
    "GA_Metadata": {
        "best_generation": params.get("best_generation", None),
        "best_auc": params.get("best_auc", None),
        "population_size": params.get("population_size", None),
        "generations": params.get("generations", None)
    }
}

RESULTS_FILE = "Prediction_Of_Sepsis_GA_Optimized_GRU_Results.json"
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n[FINAL RESULTS]")
print(json.dumps(results, indent=2))

print("\n[INFO] Training complete. Artifacts saved:")
print(f" - Model Weights : {MODEL_FILE}")
print(f" - Results JSON  : {RESULTS_FILE}")
print(f" - Scaler        : {SCALER_FILE}")
print(f" - GA Params     : {GA_PARAMS_FILE}")
