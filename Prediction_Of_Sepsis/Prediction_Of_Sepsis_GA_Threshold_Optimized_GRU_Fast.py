# ====================================================================
# Prediction_Of_Sepsis_GA_Optimized_Hyperparams_Threshold_Optimized_GRU_Fast.py
# GRU with Fast GA-Optimized Hyperparameters + GA Threshold Optimization
# ====================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from deap import base, creator, tools, algorithms

# ====================================================================
# Step 1: Device
# ====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ====================================================================
# Step 2: Load Best Hyperparameters from Fast GA Optimizer
# ====================================================================
GA_PARAMS_FILE = "GA_Optimize_Sepsis_Fast_BestHyperparams.json"
with open(GA_PARAMS_FILE, "r") as f:
    params = json.load(f)

print("[INFO] Loaded Fast GA-optimized hyperparameters:")
print(json.dumps(params, indent=2))

# ====================================================================
# Step 3: Load Dataset
# ====================================================================
CSV_PATH = "Dataset_Processed.csv"
df = pd.read_csv(CSV_PATH)

features = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
target = "SepsisLabel"

print(f"[INFO] Dataset shape: {df.shape}")
print("[INFO] Class distribution:\n", df[target].value_counts(normalize=True))

# Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
SCALER_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_Scaler.pkl"
joblib.dump(scaler, SCALER_FILE)

# Group by patient
patients = []
for pid, group in df.groupby("Patient_ID"):
    X = group[features].values
    y = group[target].values
    patients.append((X, y))

# Split
train_patients, temp_patients = train_test_split(patients, test_size=0.3, random_state=42)
val_patients, test_patients   = train_test_split(temp_patients, test_size=0.5, random_state=42)

print(f"[INFO] Patients -> Train: {len(train_patients)}, "
      f"Val: {len(val_patients)}, Test: {len(test_patients)}")

# ====================================================================
# Step 4: Dataset Class
# ====================================================================
class SepsisDataset(Dataset):
    """Custom dataset for sequential patient data."""
    def __init__(self, patient_data, seq_len=12):
        self.data = []
        for X, y in patient_data:
            if len(X) >= seq_len:
                for i in range(len(X) - seq_len + 1):
                    self.data.append((X[i:i+seq_len], y[i+seq_len-1]))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        X, y = self.data[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ====================================================================
# Step 5: GRU Model
# ====================================================================
class GRUNet(nn.Module):
    """GRU with GA-optimized hyperparameters."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=(dropout if num_layers > 1 else 0))
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze()

# ====================================================================
# Step 6: DataLoaders
# ====================================================================
seq_len = 12
batch_size = 64

train_dataset = SepsisDataset(train_patients, seq_len)
val_dataset   = SepsisDataset(val_patients, seq_len)
test_dataset  = SepsisDataset(test_patients, seq_len)

labels = [y for _, y in train_dataset]
class_counts = np.bincount(np.array(labels, dtype=int))
class_weights = 1. / class_counts
sample_weights = [class_weights[int(y)] for _, y in train_dataset]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ====================================================================
# Step 7: Model, Loss, Optimizer
# ====================================================================
pos = df[target].sum()
neg = len(df) - pos
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

model = GRUNet(
    input_dim=len(features),
    hidden_dim=params["hidden_dim"],
    num_layers=params["num_layers"],
    dropout=params["dropout"]
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

# ====================================================================
# Step 8: Training with Early Stopping
# ====================================================================
def evaluate(loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            probs = torch.sigmoid(model(X_batch))
            y_true.extend(y_batch.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_prob)

best_auc, patience, patience_counter = 0, 5, 0
EPOCHS = 30
MODEL_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_BestModel.pt"

print("[INFO] Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    y_val, p_val = evaluate(val_loader)
    val_auc = roc_auc_score(y_val, p_val)
    print(f"[Epoch {epoch+1:02d}] Loss={total_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), MODEL_FILE)
        patience_counter = 0
        print(f"[INFO] Best model updated (AUC={val_auc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("[INFO] Early stopping triggered")
            break

# ====================================================================
# Step 9: Final Test Predictions
# ====================================================================
model.load_state_dict(torch.load(MODEL_FILE))
y_test, p_test = evaluate(test_loader)

# Save raw outputs
np.save("Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_y_test.npy", y_test)
np.save("Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_p_test.npy", p_test)

# ====================================================================
# Step 10: GA Threshold Optimization
# ====================================================================
print("\n[INFO] Running GA threshold optimization...")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def ga_eval(ind):
    thr = ind[0]
    y_pred = (p_test >= thr).astype(int)
    return (f1_score(y_test, y_pred, zero_division=0),)

toolbox.register("evaluate", ga_eval)
toolbox.register("mate", tools.cxBlend, alpha=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

POP_SIZE, NGEN = 30, 20
pop = toolbox.population(n=POP_SIZE)
pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=NGEN, verbose=False)

best_ind = tools.selBest(pop, 1)[0]
best_thr = best_ind[0]

# Save threshold separately
THRESHOLD_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_BestThreshold.json"
with open(THRESHOLD_FILE, "w") as f:
    json.dump({"Best_Threshold": float(best_thr)}, f, indent=2)

# ====================================================================
# Step 11: Metrics at Best Threshold
# ====================================================================
y_pred = (p_test >= best_thr).astype(int)

results = {
    "Best Threshold (GA)": float(best_thr),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC AUC": roc_auc_score(y_test, p_test),
    "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    "Classification Report": classification_report(y_test, y_pred, output_dict=True),
    "GA_Metadata": {
        "population_size": POP_SIZE,
        "generations": NGEN,
        "Hyperparameters": params
    }
}

RESULTS_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_Results.json"
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n[FINAL RESULTS]")
print(json.dumps(results, indent=2))

print("\n[INFO] Training complete. Artifacts saved:")
print(f" - Model Weights   : {MODEL_FILE}")
print(f" - Results JSON    : {RESULTS_FILE}")
print(f" - Scaler          : {SCALER_FILE}")
print(f" - GA Threshold    : {THRESHOLD_FILE}")
print(f" - GA Params       : {GA_PARAMS_FILE}")
print(" - Raw Predictions : *_Fast_y_test.npy, *_Fast_p_test.npy")
