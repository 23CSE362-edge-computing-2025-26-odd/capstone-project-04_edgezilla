SIMULATION_DURATION = 3600
NUM_WARDS = 3
WEARABLES_PER_WARD = 5
TOTAL_WEARABLES = NUM_WARDS * WEARABLES_PER_WARD
WEARABLE_CPU = 500
WEARABLE_RAM = 256  # MB
EDGE_SERVER_CPU = 8000
CLOUD_DATACENTER_CPU = 20000
EDGE_SERVER_RAM = 32 * 1024  # MB
CLOUD_DATACENTER_RAM = 64 * 1024  # MB
LATENCY_WEARABLE_SENSOR = 6
LATENCY_WEARABLE_TO_EDGE = 5
LATENCY_EDGE_TO_CLOUD = 30
SENSOR_DATA_GENERATION_INTERVAL = 10
HEALTH_DATA_PARAMS = {
    "heart_rate": (60, 100),       # beats per minute
    "blood_oxygen": (95, 100),    # percentage
    "temperature": (36.5, 37.5),  # Celsius
    "movement": (0, 1)            # 0=idle, 1=moving
}
POWER_WEARABLE_IDLE = 0.05
POWER_WEARABLE_BUSY = 0.25
POWER_EDGE_IDLE = 200
POWER_EDGE_BUSY = 400
POWER_CLOUD_IDLE = 5000
POWER_CLOUD_BUSY = 8000

TASK_CPU_REQUIREMENT = 100
DATA_TUPLE_SIZE = 2  # KB

LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
DQN_TRAINING_INTERVAL = 50 # Train every 50 seconds of simulation time
DQN_TARGET_UPDATE_INTERVAL = 200 # Update target network every 200 seconds

# --- NEW: Experiment Configurations ---
EXPERIMENT_STRATEGIES = ['dqn', 'always_edge', 'always_cloud', 'random']
COMPARISON_METRICS = ['latency', 'throughput', 'cost'] # For future use

# --- NEW: Visualization & Export Settings ---
RESULTS_DIR = 'results'
CHART_OUTPUT_DIR = f'{RESULTS_DIR}/charts'
DATA_OUTPUT_DIR = f'{RESULTS_DIR}/data'
CHART_STYLE = 'seaborn-v0_8-darkgrid'