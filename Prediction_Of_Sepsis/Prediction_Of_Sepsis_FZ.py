# ====================================================================
# Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Fuzzy_Interpretation.py
# GRU with GA-Optimized Hyperparameters + GA Threshold + Fuzzy Logic
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
# Step 2: Load GA-Optimized Hyperparameters + Threshold
# ====================================================================
HYP_FILE = "GA_Optimize_Sepsis_BestHyperparams.json"
THR_FILE = "GA_Threshold_Optimize_Sepsis_BestThreshold.json"

with open(HYP_FILE, "r") as f:
    params = json.load(f)
with open(THR_FILE, "r") as f:
    best_thr = json.load(f)["Best_Threshold"]

print("[INFO] Loaded GA-optimized hyperparameters:")
print(json.dumps(params, indent=2))
print(f"[INFO] Loaded GA-optimized threshold: {best_thr:.4f}")

# ====================================================================
# Step 3: Load Dataset
# ====================================================================
df = pd.read_csv("Dataset_Processed.csv")
features = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]
target = "SepsisLabel"

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

patients = []
for pid, group in df.groupby("Patient_ID"):
    X = group[features].values
    y = group[target].values
    patients.append((X, y))

train_patients, temp_patients = train_test_split(patients, test_size=0.3, random_state=42)
val_patients, test_patients   = train_test_split(temp_patients, test_size=0.5, random_state=42)

print(f"[INFO] Patients -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

# ====================================================================
# Step 4: Dataset + Model
# ====================================================================
class SepsisDataset(Dataset):
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

class GRUNet(nn.Module):
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
# Step 5: Train GRU with GA Hyperparams
# ====================================================================
seq_len = 12
train_loader = DataLoader(SepsisDataset(train_patients, seq_len), batch_size=64, shuffle=True)
val_loader   = DataLoader(SepsisDataset(val_patients, seq_len), batch_size=64)
test_loader  = DataLoader(SepsisDataset(test_patients, seq_len), batch_size=64)

model = GRUNet(len(features),
               params["hidden_dim"],
               params["num_layers"],
               params["dropout"]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

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
        torch.save(model.state_dict(), "Prediction_Of_Sepsis_FZ_BestModel.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience: break

model.load_state_dict(torch.load("Prediction_Of_Sepsis_FZ_BestModel.pt"))

# ====================================================================
# Step 6: Test Predictions with Saved Threshold
# ====================================================================
y_test, p_test = evaluate(test_loader)
y_pred = (p_test >= best_thr).astype(int)

# ====================================================================
# Step 7: Fuzzy Risk Interpreter (Expanded)
# ====================================================================
print("\n[INFO] Building fuzzy risk interpreter...")

# Antecedents (inputs)
prob_in = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob')
hr_in   = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'hr_z')
o2_in   = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'o2_z')
temp_in = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'temp_z')
resp_in = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'resp_z')
map_in  = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'map_z')

# Consequent (output risk)
risk_out = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'risk')

# Membership functions
for ant in [prob_in, hr_in, o2_in, temp_in, resp_in, map_in]:
    ant['low']  = fuzz.trimf(ant.universe, [ant.universe.min(), ant.universe.min(), 0])
    ant['norm'] = fuzz.trimf(ant.universe, [-1, 0, 1])
    ant['high'] = fuzz.trimf(ant.universe, [0, ant.universe.max(), ant.universe.max()])

risk_out['low']  = fuzz.trimf(risk_out.universe, [0, 0, 0.5])
risk_out['med']  = fuzz.trimf(risk_out.universe, [0.25, 0.5, 0.75])
risk_out['high'] = fuzz.trimf(risk_out.universe, [0.5, 1, 1])

# Rules (explanatory)
rules = [
    ctrl.Rule(prob_in['high'] & hr_in['high'], risk_out['high']),
    ctrl.Rule(prob_in['high'] & o2_in['low'], risk_out['high']),
    ctrl.Rule(prob_in['high'] & temp_in['high'], risk_out['high']),
    ctrl.Rule(prob_in['med'] & resp_in['high'], risk_out['med']),
    ctrl.Rule(prob_in['low'] & map_in['norm'], risk_out['low']),
    ctrl.Rule(prob_in['low'], risk_out['low'])
]

risk_ctrl = ctrl.ControlSystem(rules)
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

def fuzzy_explain(prob, hr_z, o2_z, temp_z, resp_z, map_z):
    risk_sim.input['prob'] = prob
    risk_sim.input['hr_z'] = hr_z
    risk_sim.input['o2_z'] = o2_z
    risk_sim.input['temp_z'] = temp_z
    risk_sim.input['resp_z'] = resp_z
    risk_sim.input['map_z'] = map_z
    risk_sim.compute()
    risk_val = risk_sim.output['risk']
    explanation = []
    if prob > 0.7: explanation.append("High GRU probability")
    if hr_z > 1: explanation.append("Elevated HR")
    if o2_z < -1: explanation.append("Low O2Sat")
    if temp_z > 1: explanation.append("Fever (High Temp)")
    if resp_z > 1: explanation.append("High Respiration")
    if map_z < -1: explanation.append("Low MAP")
    return risk_val, " + ".join(explanation) if explanation else "Normal range vitals"

# ====================================================================
# Step 8: Collect Results with Fuzzy Explanations
# ====================================================================
fuzzy_results = []
for i in range(len(y_test)):
    if y_pred[i] == 1:  # Only explain sepsis positives
        hr_z, o2_z, temp_z, map_z, resp_z = test_patients[i][0][0], test_patients[i][0][1], test_patients[i][0][2], test_patients[i][0][4], test_patients[i][0][6]
        risk_val, explanation = fuzzy_explain(p_test[i], hr_z, o2_z, temp_z, resp_z, map_z)
        fuzzy_results.append({
            "Sample": i,
            "Probability": float(p_test[i]),
            "Threshold": float(best_thr),
            "Predicted_Label": int(y_pred[i]),
            "FuzzyRisk": float(risk_val),
            "FuzzyLevel": "HIGH" if risk_val > 0.7 else "MED" if risk_val > 0.4 else "LOW",
            "FuzzyExplanation": explanation
        })

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
    "Fuzzy_Interpretations": fuzzy_results
}

RESULTS_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Fuzzy_Results.json"
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n[FINAL RESULTS with FUZZY INTERPRETATION]")
print(json.dumps(results, indent=2))
