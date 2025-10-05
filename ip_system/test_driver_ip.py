import requests
import numpy as np
import time
import random
import os

EDGE_URL = "http://127.0.0.1:5002"
CLOUD_URL = "http://127.0.0.1:5003"
NUM_EDGES = 3
EDGE_IDS = [f"edge_{i+1}" for i in range(NUM_EDGES)]
SIMULATION_STEPS = 50
CLOUD_AGGREGATION_INTERVAL = 5

def generate_patient_state():
    """Generates realistic patient vital signs for sepsis detection."""
    hr = random.uniform(50, 150)
    temp = random.uniform(36.0, 40.0)
    bp = random.uniform(80, 180)
    resp = random.uniform(10, 30)
    o2sat = random.uniform(90, 100)
    glucose = random.uniform(70, 200)
    is_sepsis_likely = hr > 100 or temp > 38.5 or bp < 90 or resp > 22
    return np.array([hr, temp, bp, resp, o2sat, glucose]), is_sepsis_likely

def simulate_inference(action, edge_cpu_load):
    """Simulates inference execution based on action choice."""
    if action == 0:  # Edge processing
        latency = 0.5 + edge_cpu_load * 2.0 + random.uniform(-0.1, 0.1)
        cpu_util = min(0.95, edge_cpu_load + 0.3 + random.uniform(-0.05, 0.05))
        queue_length = max(0, int(edge_cpu_load * 5) - 1)
    else:  # Cloud offloading
        latency = 0.2 + random.uniform(-0.05, 0.05)
        cpu_util = max(0.1, edge_cpu_load - 0.2 + random.uniform(-0.05, 0.05))
        queue_length = 0
    return latency, cpu_util, queue_length

def print_header(title):
    """Prints formatted section header."""
    print("\n" + "="*80)
    print(f"====== {title.center(68)} ======")
    print("="*80)

def print_edge_decision(step, edge_id, state, action, state_value, is_explore):
    """Prints edge decision information."""
    action_str = "PROCESS ON EDGE" if action == 0 else "OFFLOAD TO CLOUD"
    explore_str = " (Explore)" if is_explore else " (Exploit)"
    print(f"[{step}] EDGE IP: {edge_id}")
    print(f"  - Patient State : HR={state[0]:.1f}, Temp={state[1]:.1f}, BP={state[2]:.1f}, RR={state[3]:.1f}, O2={state[4]:.1f}, Glu={state[5]:.1f}")
    print(f"  - State Value   : {state_value:.3f}")
    print(f"  - Decision      : {action_str}{explore_str}")

def print_edge_result(latency, cpu, queue, base_reward, loss):
    """Prints edge execution results."""
    print(f"  - Outcome       : Latency={latency:.2f}s, Edge CPU={cpu*100:.1f}%, Queue={queue}")
    print(f"  - Sepsis Reward : {base_reward:.2f}")
    print(f"  - Stored & Trained (Loss: {loss:.4f})\n")

def print_cloud_update(data):
    """Prints cloud coordination update."""
    strategy = data['allocation_strategy']['priority']
    reward = data['cloud_reward']
    dist_rewards = data['distributed_rewards']
    print(f"CLOUD IP: Aggregated state values from {len(EDGE_IDS)} edges.")
    print(f"  - Cloud Decision : New resource strategy is '{strategy.upper()}'")
    print(f"  - Global Reward  : Calculated {reward:.3f} based on multi-objective optimization.")
    print(f"  - Distributing Rewards to Edges:")
    for edge_id, r in dist_rewards.items():
        print(f"    - {edge_id}: {r:.3f}")

if __name__ == "__main__":
    edge_cpu_loads = {edge_id: random.uniform(0.2, 0.5) for edge_id in EDGE_IDS}
    edge_contributions = {edge_id: 0 for edge_id in EDGE_IDS}
    
    print_header("STARTING EDGE-CLOUD IP SEPSIS DETECTION SIMULATION")
    time.sleep(2)

    for step in range(1, SIMULATION_STEPS + 1):
        print_header(f"SIMULATION STEP {step}")
        for edge_id in EDGE_IDS:
            try:
                current_state, is_sepsis_likely = generate_patient_state()

                # Get action from IP agent
                res = requests.post(f"{EDGE_URL}/get_action", json={'edge_id': edge_id, 'state': current_state.tolist()})
                res.raise_for_status()
                data = res.json()
                action, epsilon, state_value, is_explore = data['action'], data['epsilon'], data['state_value'], data['is_explore']
                
                print_edge_decision(step, edge_id, current_state, action, state_value, is_explore)

                # Simulate inference execution
                latency, cpu_util, queue_length = simulate_inference(action, edge_cpu_loads[edge_id])
                
                # Calculate base reward
                base_reward = 10.0 if is_sepsis_likely else -1.0
                
                # Generate next state and store transition
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

                # Update edge state
                edge_cpu_loads[edge_id] = cpu_util
                edge_contributions[edge_id] += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"\n[ERROR] Could not connect to Edge IP server at {EDGE_URL}. Is it running?")
                print(f"  Details: {e}")
                exit()
        
        # Cloud coordination every few steps
        if step % CLOUD_AGGREGATION_INTERVAL == 0:
            print_header(f"CLOUD COORDINATION @ STEP {step}")
            try:
                # Get state values from edge agents
                res = requests.post(f"{EDGE_URL}/get_state_values", json={'edge_ids': EDGE_IDS})
                res.raise_for_status()
                edge_state_values = res.json()['state_values']

                # Generate system metrics
                system_metrics = {
                    'cpu_util': random.uniform(0.6, 0.9), 'memory_util': random.uniform(0.5, 0.8),
                    'queue_length': random.randint(5, 20), 'throughput': random.uniform(50, 100),
                    'fairness_index': random.uniform(0.8, 0.98),
                    'edge_contributions': edge_contributions
                }

                # Send to cloud for coordination
                payload = {'edge_state_values': edge_state_values, 'system_metrics': system_metrics}
                res = requests.post(f"{CLOUD_URL}/aggregate_and_allocate", json=payload)
                res.raise_for_status()
                cloud_data = res.json()
                print_cloud_update(cloud_data)
                
                # Distribute global rewards to edges
                for edge_id, global_reward in cloud_data['distributed_rewards'].items():
                    requests.post(f"{EDGE_URL}/set_global_reward", json={'edge_id': edge_id, 'global_reward': global_reward})

                # Reset contributions
                edge_contributions = {edge_id: 0 for edge_id in EDGE_IDS}
                
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"\n[ERROR] Could not connect to a server during cloud update. Are both servers running?")
                print(f"  Details: {e}")
                exit()

    print_header("IP SIMULATION COMPLETE")
