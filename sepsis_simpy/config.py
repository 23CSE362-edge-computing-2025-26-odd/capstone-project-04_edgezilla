# ===================================================================
# --- Simulation Basics ---
# ===================================================================
SIMULATION_DURATION = 60  # 1 minute simulation
NUM_WARDS = 3
WEARABLES_PER_WARD = 5
TOTAL_WEARABLES = NUM_WARDS * WEARABLES_PER_WARD

# ===================================================================
# --- Device Compute Specifications (MIPS-equivalent realistic setup) ---
# ===================================================================

# Wearable (IoT node / sensor device)
WEARABLE_CPU = 500           # MIPS
WEARABLE_RAM = 256           # MB

# Edge Server (local gateway / ward-level edge)
# Matches iFogSim: 4 PEs × 8000 MIPS = 32000 total
EDGE_SERVER_CPU = 32000      # MIPS
EDGE_SERVER_RAM = 32 * 1024  # 32 GB

# Cloud Datacenter (remote cloud)
# Matches iFogSim: 8 PEs × 20000 MIPS = 160000 total
CLOUD_DATACENTER_CPU = 160000  # MIPS
CLOUD_DATACENTER_RAM = 64 * 1024  # 64 GB

# ===================================================================
# --- Network Latency Configuration (realistic RTTs) ---
# ===================================================================

# Local sensor reading
LATENCY_WEARABLE_SENSOR = 6       # ms

# Wireless hop from wearable to edge (Wi-Fi / BLE)
LATENCY_WEARABLE_TO_EDGE = 5      # ms

# Edge to Cloud latency (WAN / backbone RTT)
# Adjusted for realistic ~73 ms overall E2E latency
LATENCY_EDGE_TO_CLOUD = 60        # ms typical RTT
# (If simulating slower internet, you can increase to 100 ms)

# ===================================================================
# --- Task / Data Characteristics ---
# ===================================================================

SENSOR_DATA_GENERATION_INTERVAL = 7  # seconds between health data tuples

HEALTH_DATA_PARAMS = {
    "heart_rate": (60, 100),        # bpm
    "blood_oxygen": (95, 100),      # %
    "temperature": (36.5, 37.5),    # °C
    "movement": (0, 1)              # 0=idle, 1=active
}

# Typical lightweight processing (e.g., feature extraction/classification)
# Task CPU requirements (MIPS)
TASK_CPU_REQUIREMENT = 100  # MIPS per task

# ML Inference Time (seconds)
# Realistic ML inference time without HTTP overhead
ML_INFERENCE_TIME = 0.015  # 15ms for sepsis detection model → yields ~12 ms (edge) & ~73 ms (cloud)
DATA_TUPLE_SIZE = 4  # KB (health packet size)

# ===================================================================
# --- Power and Energy Models (W) ---
# ===================================================================

POWER_WEARABLE_IDLE = 1.0
POWER_WEARABLE_BUSY = 3.0

POWER_EDGE_SERVER_IDLE = 100.0
POWER_EDGE_SERVER_BUSY = 300.0

POWER_CLOUD_INSTANCE_IDLE = 200.0
POWER_CLOUD_INSTANCE_BUSY = 500.0

# Energy cost for network transmission (J/MB)
ENERGY_PER_MB_WIRELESS = 1.8   # Wearable → Edge
ENERGY_PER_MB_WIRED = 0.5      # Edge → Cloud

HEALTH_DATA_PACKET_SIZE_KB = 4.0  # For transmission energy computation

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
