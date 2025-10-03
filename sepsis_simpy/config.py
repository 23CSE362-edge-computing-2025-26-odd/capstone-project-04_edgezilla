# config.py (Enhanced for Performance Monitoring)

# --- Existing Configurations ---
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
POWER_WEARABLE_BUSY = 0.25
DATA_TUPLE_SIZE = 2  # KB

TASK_CPU_REQUIREMENT = 100

# --- DQN Hyperparameters ---
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
DQN_TRAINING_INTERVAL = 50
DQN_TARGET_UPDATE_INTERVAL = 200

# --- Experiment Configurations ---
EXPERIMENT_STRATEGIES = ['dqn', 'always_edge', 'always_cloud', 'random']

# --- Visualization & Export Settings ---
RESULTS_DIR = 'results'
CHART_OUTPUT_DIR = f'{RESULTS_DIR}/charts'
DATA_OUTPUT_DIR = f'{RESULTS_DIR}/data'
CHART_STYLE = 'seaborn-v0_8-darkgrid'

# ===================================================================
# NEW: Performance, Energy, and Accuracy Configurations
# ===================================================================

# --- Power Consumption Models (in Watts) ---
# Values are illustrative and should be based on hardware datasheets
POWER_WEARABLE_IDLE = 0.05
POWER_WEARABLE_TRANSMIT = 0.35  # Energy for sending data
POWER_EDGE_SERVER_IDLE = 150.0
POWER_EDGE_SERVER_BUSY = 400.0   # When processing a task
POWER_CLOUD_INSTANCE_IDLE = 300.0
POWER_CLOUD_INSTANCE_BUSY = 650.0

# --- Energy for Network Transmission (Joules per Megabyte) ---
# This simplifies calculating network energy costs
ENERGY_PER_MB_WIRED = 0.5   # Edge to Cloud
ENERGY_PER_MB_WIRELESS = 1.8 # Wearable to Edge

# --- Data Sizes (in Kilobytes) ---
HEALTH_DATA_PACKET_SIZE_KB = 4.0

# --- Accuracy Evaluation ---
# Defines a simple "optimal" policy for comparison
# If CPU utilization exceeds this, the "correct" decision is to offload
CPU_UTIL_THRESHOLD_FOR_OFFLOAD = 0.80
# If the queue is longer than this, the "correct" decision is to offload
QUEUE_LENGTH_THRESHOLD_FOR_OFFLOAD = 10

# --- Reporting ---
PERFORMANCE_REPORT_FILENAME = 'performance_summary_report.txt'
PERFORMANCE_DATA_FILENAME = 'detailed_performance_metrics.json'
