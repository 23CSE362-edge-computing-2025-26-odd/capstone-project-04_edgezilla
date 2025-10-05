# ===================================================================
# --- Simulation Basics ---
# ===================================================================
SIMULATION_DURATION = 60  # seconds
NUM_WARDS = 3
WEARABLES_PER_WARD = 5
TOTAL_WEARABLES = NUM_WARDS * WEARABLES_PER_WARD

# ===================================================================
# --- Device Compute Specifications (MIPS-equivalent realistic setup) ---
# ===================================================================

# Wearable (IoT node)
WEARABLE_CPU = 500
WEARABLE_RAM = 256  # MB

# Edge Server (local gateway)
EDGE_SERVER_CPU = 32000       # MIPS
EDGE_SERVER_RAM = 32 * 1024   # 32 GB

# Cloud Datacenter (remote)
CLOUD_DATACENTER_CPU = 160000 # MIPS
CLOUD_DATACENTER_RAM = 64 * 1024  # 64 GB

# ===================================================================
# --- Network Latency Configuration (realistic RTTs) ---
# ===================================================================

LATENCY_WEARABLE_SENSOR = 8          # ms (sensor sampling + preproc)
LATENCY_WEARABLE_TO_EDGE = 10        # ms (BLE/Wi-Fi hop)
LATENCY_EDGE_TO_CLOUD = 300          # ms  <<< increased (realistic WAN + HTTP overhead)

# ===================================================================
# --- Task / Data Characteristics ---
# ===================================================================

SENSOR_DATA_GENERATION_INTERVAL = 7  # seconds

HEALTH_DATA_PARAMS = {
    "heart_rate": (60, 100),
    "blood_oxygen": (95, 100),
    "temperature": (36.5, 37.5),
    "movement": (0, 1)
}

# ===================================================================
# --- Processing Requirements (adjusted for stronger contrast) ---
# ===================================================================

# Edge: ~130–160 ms target
#   32 000 MIPS × 0.14 s ≈ 4 480 M instructions
# Cloud: ~330 ms total (180 ms network + ~150 ms inference)
#   160 000 MIPS × 0.045 s ≈ 7 200 M instructions
TASK_CPU_REQUIREMENT = 4500  # MIPS-equivalent per task

DATA_TUPLE_SIZE = 4  # KB

# ===================================================================
# --- Power and Energy Models (W) ---
# ===================================================================

POWER_WEARABLE_IDLE = 1.0
POWER_WEARABLE_BUSY = 3.0

POWER_EDGE_SERVER_IDLE = 120.0
POWER_EDGE_SERVER_BUSY = 320.0

POWER_CLOUD_INSTANCE_IDLE = 250.0
POWER_CLOUD_INSTANCE_BUSY = 550.0

ENERGY_PER_MB_WIRELESS = 1.8
ENERGY_PER_MB_WIRED = 0.5
HEALTH_DATA_PACKET_SIZE_KB = 4.0

# ===================================================================
# --- DQN / Learning Parameters (unchanged) ---
# ===================================================================

LEARNING_RATE = 2e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 128
MEMORY_CAPACITY = 20000
DQN_TRAINING_INTERVAL = 25
DQN_TARGET_UPDATE_INTERVAL = 100

# ===================================================================
# --- Offloading / Accuracy Thresholds ---
# ===================================================================

CPU_UTIL_THRESHOLD_FOR_OFFLOAD = 0.80
QUEUE_LENGTH_THRESHOLD_FOR_OFFLOAD = 10

# ===================================================================
# --- Experiment / Output Configuration ---
# ===================================================================

EXPERIMENT_STRATEGIES = ['dqn', 'always_edge', 'always_cloud', 'random']

RESULTS_DIR = 'results'
CHART_OUTPUT_DIR = f'{RESULTS_DIR}/charts'
DATA_OUTPUT_DIR = f'{RESULTS_DIR}/data'
CHART_STYLE = 'seaborn-v0_8-darkgrid'

PERFORMANCE_REPORT_FILENAME = 'performance_summary_report.txt'
PERFORMANCE_DATA_FILENAME = 'detailed_performance_metrics.json'
