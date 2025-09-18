import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ================================================================
# 1. Load Model, Scaler, and Threshold
# ================================================================
MODEL_FILE = "Prediction_Of_Sepsis_FZ_Fast_BestModel.pt"
SCALER_FILE = "Prediction_Of_Sepsis_GA_Optimized_GRU_Fast_Scaler.pkl"
THRESHOLD_FILE = "Prediction_Of_Sepsis_GA_Hyperparam_Threshold_Optimized_Fast_BestThreshold.json"

with open(THRESHOLD_FILE, "r") as f:
    best_thr = json.load(f)["Best_Threshold"]

scaler = joblib.load(SCALER_FILE)

# GRU model (same structure used before)
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

# Load params
with open("GA_Optimize_Sepsis_Fast_BestHyperparams.json") as f:
    params = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRUNet(8, params["hidden_dim"], params["num_layers"], params["dropout"]).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

# ================================================================
# 2. Build Fuzzy Interpreter
# ================================================================
prob_in = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob')
hr_in   = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'hr_z')
o2_in   = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'o2_z')
temp_in = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'temp_z')
resp_in = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'resp_z')
map_in  = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'map_z')
risk_out = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'risk')

# Membership functions
prob_in['low']  = fuzz.trimf(prob_in.universe, [prob_in.universe.min(), prob_in.universe.min(), 0])
prob_in['norm'] = fuzz.trimf(prob_in.universe, [-1, 0, 1])
prob_in['high'] = fuzz.trimf(prob_in.universe, [0, prob_in.universe.max(), prob_in.universe.max()])
prob_in['med']  = fuzz.trimf(prob_in.universe, [0, 0.5, 1])  # Added 'med' membership function

for ant in [hr_in, o2_in, temp_in, resp_in, map_in]:
    ant['low']  = fuzz.trimf(ant.universe, [ant.universe.min(), ant.universe.min(), 0])
    ant['norm'] = fuzz.trimf(ant.universe, [-1, 0, 1])
    ant['high'] = fuzz.trimf(ant.universe, [0, ant.universe.max(), ant.universe.max()])

risk_out['low']  = fuzz.trimf(risk_out.universe, [0, 0, 0.5])
risk_out['med']  = fuzz.trimf(risk_out.universe, [0.25, 0.5, 0.75])
risk_out['high'] = fuzz.trimf(risk_out.universe, [0.5, 1, 1])

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
    if temp_z > 1: explanation.append("Fever")
    if resp_z > 1: explanation.append("High Respiration")
    if map_z < -1: explanation.append("Low MAP")
    return risk_val, " + ".join(explanation) if explanation else "Normal range"

# ================================================================
# 3. Real-Time Simulation
# ================================================================
features = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]

# Option 1: Random Simulation
def random_simulation(times):
    print("\n[SIMULATION TYPE] Random Simulation\n")
    for i in range(times):  # Run the simulation as many times as specified
        time.sleep(1)
        case = {
            "HR": np.random.randint(60, 180),
            "O2Sat": np.random.randint(70, 100),
            "Temp": np.random.uniform(35.0, 42.0),
            "SBP": np.random.randint(80, 150),
            "MAP": np.random.randint(50, 110),
            "DBP": np.random.randint(30, 100),
            "Resp": np.random.randint(12, 35),
            "EtCO2": np.random.uniform(10, 50)
        }
        
        # Ensure you're using a DataFrame with the correct columns
        x = pd.DataFrame([case])[features]
        
        # Transform the input using the fitted scaler
        x_scaled = scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        prob = torch.sigmoid(model(x_tensor)).item()
        pred = int(prob >= best_thr)

        # Fuzzy interpretation
        z_vals = x_scaled[0]  # scaled vitals
        risk_val, explanation = fuzzy_explain(prob, z_vals[0], z_vals[1], z_vals[2],
                                              z_vals[6], z_vals[4])

        print(f"   Time {i+1}s | Input={case}")
        print(f"   GRU Prob={prob:.4f}, Threshold={best_thr:.4f}, Pred={pred}")
        print(f"   Risk={risk_val:.3f}, Level={'HIGH' if risk_val>0.7 else 'MED' if risk_val>0.4 else 'LOW'}")
        print(f"   Explanation: {explanation}\n")

# ================================================================
# 4. Choose Simulation Type
# ================================================================
def run_simulation():
    print("Choose Simulation Type:")
    print("1. Random Simulation")
    choice = input("Enter 1: ")

    if choice == '1':
        times = int(input("Enter the number of times to run simulation (1-10): "))
        random_simulation(times)
    else:
        print("Invalid choice. Exiting simulation.")

run_simulation()
