# ====================================================================
# GA_Optimize_Sepsis_Fast.py
# Fast Genetic Algorithm-based Hyperparameter Optimization for GRU
# (Population=10, Generations=5)
# ====================================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from deap import base, creator, tools, algorithms

# ====================================================================
# Step 1: Config + Logging
# ====================================================================
BASE_NAME = "GA_Optimize_Sepsis_Fast"
BEST_PARAMS_FILE = f"{BASE_NAME}_BestHyperparams.json"
LOG_FILE = f"{BASE_NAME}_TrainingLog.txt"

def log_message(message: str, end: str = "\n"):
    print(message, end=end)
    with open(LOG_FILE, "a") as f:
        f.write(message + end)

# Fixed smaller pop & gen
POP_SIZE = 10
NGEN = 5

# Reset log file
open(LOG_FILE, "w").close()
log_message("=" * 70)
log_message(f" Genetic Algorithm Hyperparameter Optimization ({BASE_NAME}) ")
log_message("=" * 70)
log_message(f"[INFO] Population Size = {POP_SIZE}, Generations = {NGEN}")

# ====================================================================
# Step 2: Dataset Prep
# ====================================================================
df = pd.read_csv("Dataset_Processed.csv")
features = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
target = "SepsisLabel"

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

patients = []
for pid, group in df.groupby("Patient_ID"):
    X = group[features].values
    y = group[target].values
    patients.append((X, y))

train_patients, val_patients = train_test_split(patients, test_size=0.2, random_state=42)

log_message(f"[INFO] Dataset loaded: {df.shape} rows, {len(patients)} patients")
log_message(f"[INFO] Train: {len(train_patients)}, Validation: {len(val_patients)}")

# ====================================================================
# Step 3: Dataset + Model
# ====================================================================
class SepsisDataset(Dataset):
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
# Step 4: GA Evaluation
# ====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
log_message(f"[INFO] Using computation device: {device}")

def evaluate_individual(individual):
    hidden_dim, num_layers, dropout, lr = individual
    hidden_dim, num_layers = int(hidden_dim), int(num_layers)

    train_dataset = SepsisDataset(train_patients)
    val_dataset = SepsisDataset(val_patients)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = GRUNet(len(features), hidden_dim, num_layers, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Very light training
    model.train()
    for epoch in range(2):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = torch.sigmoid(model(X_batch))
            y_true.extend(y_batch.numpy())
            y_prob.extend(probs.cpu().numpy())

    return (roc_auc_score(y_true, y_prob),)

# ====================================================================
# Step 5: GA Setup
# ====================================================================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("hidden_dim", random.choice, [64, 128, 256])
toolbox.register("num_layers", random.choice, [1, 2, 3])
toolbox.register("dropout", random.choice, [0.2, 0.3, 0.5])
toolbox.register("lr", random.choice, [0.001, 0.0005, 0.0001])

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.hidden_dim, toolbox.num_layers,
                  toolbox.dropout, toolbox.lr), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)

# ✅ Safe custom mutation
def custom_mutate(individual):
    if random.random() < 0.2:
        individual[0] = random.choice([64, 128, 256])
    if random.random() < 0.2:
        individual[1] = random.choice([1, 2, 3])
    if random.random() < 0.2:
        individual[2] = random.choice([0.2, 0.3, 0.5])
    if random.random() < 0.2:
        individual[3] = random.choice([0.001, 0.0005, 0.0001])
    return (individual,)

toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# ====================================================================
# Step 6: Run GA
# ====================================================================
pop = toolbox.population(n=POP_SIZE)

log_message("[GA] Starting Fast GA optimization...")

best_individual, best_auc, best_generation = None, -1.0, -1

for gen in range(1, NGEN + 1):
    log_message(f"\n[GA] --- Generation {gen} ---")
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
    fits = list(map(toolbox.evaluate, offspring))

    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit

    pop[:] = toolbox.select(offspring, k=len(pop))

    gen_best = tools.selBest(pop, k=1)[0]
    gen_auc = gen_best.fitness.values[0]
    log_message(f"[GA] Best individual (Gen {gen}): {gen_best}, AUC={gen_auc:.4f}")

    if gen_auc > best_auc:
        best_auc, best_individual, best_generation = gen_auc, gen_best, gen

# ====================================================================
# Step 7: Save Results
# ====================================================================
log_message(f"\n[RESULT] Best hyperparameters from Gen {best_generation}: {best_individual}, AUC={best_auc:.4f}")

with open(BEST_PARAMS_FILE, "w") as f:
    json.dump({
        "hidden_dim": int(best_individual[0]),
        "num_layers": int(best_individual[1]),
        "dropout": float(best_individual[2]),
        "lr": float(best_individual[3]),
        "best_generation": best_generation,
        "best_auc": best_auc,
        "population_size": POP_SIZE,
        "generations": NGEN
    }, f, indent=2)

log_message(f"[INFO] Saved best hyperparameters to {BEST_PARAMS_FILE}")
