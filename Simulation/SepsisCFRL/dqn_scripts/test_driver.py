import requests
import numpy as np
import time
import random
import os

EDGE_URL = "http://127.0.0.1:5000"
CLOUD_URL = "http://127.0.0.1:5001"
NUM_EDGES = 3
EDGE_IDS = [f"edge_{i+1}" for i in range(NUM_EDGES)]
SIMULATION_STEPS = 50
CLOUD_AGGREGATION_INTERVAL = 5

def generate_patient_state():
    hr = random.uniform(50, 150)
    temp = random.uniform(36.0, 40.0)
    bp = random.uniform(80, 180)
    resp = random.uniform(10, 30)
    o2sat = random.uniform(90, 100)
    glucose = random.uniform(70, 200)
    is_sepsis_likely = hr > 100 or temp > 38.5 or bp < 90 or resp > 22
    return np.array([hr, temp, bp, resp, o2sat, glucose]), is_sepsis_likely

def simulate_inference(action, edge_cpu_load):
    if action == 0:
        latency = 0.5 + edge_cpu_load * 2.0 + random.uniform(-0.1, 0.1)
        cpu_util = min(0.95, edge_cpu_load + 0.3 + random.uniform(-0.05, 0.05))
        queue_length = max(0, int(edge_cpu_load * 5) - 1)
    else:
        latency = 0.2 + random.uniform(-0.05, 0.05)
        cpu_util = max(0.1, edge_cpu_load - 0.2 + random.uniform(-0.05, 0.05))
        queue_length = 0
    return latency, cpu_util, queue_length

def print_header(title):
    print("\n" + "="*80)
    print(f"====== {title.center(68)} ======")
    print("="*80)

def print_edge_decision(step, edge_id, state, q_vals, action, is_explore):
    action_str = "PROCESS ON EDGE" if action == 0 else "OFFLOAD TO CLOUD"
    explore_str = " (Explore)" if is_explore else " (Exploit)"
    print(f"[{step}] EDGE SIM: {edge_id}")
    print(f"  - Patient State : HR={state[0]:.1f}, Temp={state[1]:.1f}, BP={state[2]:.1f}, RR={state[3]:.1f}, O2={state[4]:.1f}, Glu={state[5]:.1f}")
    print(f"  - DQN Q-Values  : Edge={q_vals[0]:.3f}, Cloud={q_vals[1]:.3f}")
    print(f"  - Decision      : {action_str}{explore_str}")

def print_edge_result(latency, cpu, queue, base_reward, loss):
    print(f"  - Outcome       : Latency={latency:.2f}s, Edge CPU={cpu*100:.1f}%, Queue={queue}")
    print(f"  - Sepsis Reward : {base_reward:.2f}")
    print(f"  - Stored & Trained (Loss: {loss:.4f})\n")

if __name__ == "__main__":
    edge_cpu_loads = {edge_id: random.uniform(0.2, 0.5) for edge_id in EDGE_IDS}
    edge_contributions = {edge_id: 0 for edge_id in EDGE_IDS}
    
    print_header("STARTING EDGE-CLOUD DQN SEPSIS DETECTION SIMULATION")
    time.sleep(2)

    for step in range(1, SIMULATION_STEPS + 1):
        print_header(f"SIMULATION STEP {step}")
        for edge_id in EDGE_IDS:
            try:
                current_state, is_sepsis_likely = generate_patient_state()

                res = requests.post(f"{EDGE_URL}/get_q_values", json={'edge_id': edge_id, 'state': current_state.tolist()})
                res.raise_for_status()
                data = res.json()
                q_values, epsilon = np.array(data['q_values']), data['epsilon']

                is_exploratory = False
                if random.random() < epsilon:
                    action = random.choice([0, 1])
                    is_exploratory = True
                else:
                    action = np.argmax(q_values)
                
                print_edge_decision(step, edge_id, current_state, q_values, action, is_exploratory)

                latency, cpu_util, queue_length = simulate_inference(action, edge_cpu_loads[edge_id])
                
                base_reward = 10.0 if is_sepsis_likely else -1.0
                
                next_state, _ = generate_patient_state()
                payload = {
                    'edge_id': edge_id, 'state': current_state.tolist(), 'action': action,
                    'reward': base_reward, 'next_state': next_state.tolist(), 'done': False,
                    'latency': latency, 'cpu_util': cpu_util, 'queue_length': queue_length
                }
                res = requests.post(f"{EDGE_URL}/store_transition", json=payload)
                res.raise_for_status()
                loss = res.json().get('loss', 0)
                
                print_edge_result(latency, cpu_util, queue_length, base_reward, loss)

                edge_cpu_loads[edge_id] = cpu_util
                edge_contributions[edge_id] += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"\n[ERROR] Could not connect to Edge server at {EDGE_URL}. Is it running?")
                print(f"  Details: {e}")
                exit()

    print_header("SIMULATION COMPLETE")