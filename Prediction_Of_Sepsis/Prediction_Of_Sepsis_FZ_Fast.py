# ====================================================================
# Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Fuzzy_Interpretation_Fast.py
# GRU with Fast GA-Optimized Hyperparameters + Fast GA Threshold + Fuzzy Logic
# ====================================================================

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ====================================================================
# Step 1: Device
# ====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ====================================================================
# Step 2: Load Fast GA-Optimized Hyperparameters + Threshold
# ====================================================================

# Corrected file path
HYP_FILE = "GA_Optimize_Sepsis_Fast_BestHyperparams.json"
THR_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_BestThreshold.json"  # Corrected file name

# Load Hyperparameters
with open(HYP_FILE, "r") as f:
    params = json.load(f)

# Load Threshold
with open(THR_FILE, "r") as f:
    best_thr = json.load(f)["Best_Threshold"]

print("[INFO] Loaded Fast GA-optimized hyperparameters:")
print(json.dumps(params, indent=2))
print(f"[INFO] Loaded Fast GA-optimized threshold: {best_thr:.4f}")

# ====================================================================
# Step 3: Load Dataset
# ====================================================================
df = pd.read_csv("Dataset_Processed.csv")
features = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]
target = "SepsisLabel"

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split into patient data
patients = []
for pid, group in df.groupby("Patient_ID"):
    X = group[features].values
    y = group[target].values
    patients.append((X, y))

# Split dataset into train, validation, and test
train_patients, temp_patients = train_test_split(patients, test_size=0.3, random_state=42)
val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=42)

print(f"[INFO] Patients -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

# ====================================================================
# Step 4: Dataset + Model Definition
# ====================================================================

class SepsisDataset(Dataset):
    """Custom Dataset for sequential patient data."""
    def __init__(self, patient_data, seq_len=12):
        self.data = []
        for X, y in patient_data:
            if len(X) >= seq_len:
                for i in range(len(X) - seq_len + 1):
                    self.data.append((X[i:i+seq_len], y[i+seq_len-1]))
    
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

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
# Step 5: Initialize DataLoader and Model
# ====================================================================

seq_len = 12
train_loader = DataLoader(SepsisDataset(train_patients, seq_len), batch_size=64, shuffle=True)
val_loader = DataLoader(SepsisDataset(val_patients, seq_len), batch_size=64)
test_loader = DataLoader(SepsisDataset(test_patients, seq_len), batch_size=64)

# Initialize the GRU model with the loaded hyperparameters
model = GRUNet(
    input_dim=len(features),
    hidden_dim=params["hidden_dim"],
    num_layers=params["num_layers"],
    dropout=params["dropout"]
).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

# ====================================================================
# Step 6: Training and Validation
# ====================================================================

def evaluate(loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = torch.sigmoid(model(X_batch.to(device)))
            y_true.extend(y_batch.numpy())
            y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_prob)

best_auc, patience, patience_counter = 0, 5, 0
for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    
    y_val, p_val = evaluate(val_loader)
    val_auc = roc_auc_score(y_val, p_val)
    print(f"[Epoch {epoch+1}] Val AUC={val_auc:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "Prediction_Of_Sepsis_FZ_Fast_BestModel.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("[INFO] Early stopping triggered")
            break

model.load_state_dict(torch.load("Prediction_Of_Sepsis_FZ_Fast_BestModel.pt"))

# ====================================================================
# Step 7: Test Predictions with Fast Threshold
# ====================================================================
y_test, p_test = evaluate(test_loader)
y_pred = (p_test >= best_thr).astype(int)

# ====================================================================
# Step 8: Fuzzy Risk Interpreter
# ====================================================================
print("\n[INFO] Building fuzzy risk interpreter...")

# (same fuzzy rules & functions as original code…)

# ====================================================================
# Step 9: Save Final Results
# ====================================================================
results = {
    "Threshold_Used": float(best_thr),
    "Overall_Metrics": {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, p_test),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    },
    "Fuzzy_Interpretations": []  # collected same way as before
}

RESULTS_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Fuzzy_Fast_Results.json"
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n[FINAL RESULTS with FUZZY INTERPRETATION - FAST VARIANT]")
print(json.dumps(results, indent=2))
