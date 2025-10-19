# ===================================================================
# --- Simulation Basics ---
# ===================================================================
SIMULATION_DURATION = 60 # seconds
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
# --- Enhanced DQN / Learning Parameters ---
# ===================================================================

# Core learning parameters - enhanced for deeper networks
LEARNING_RATE = 1e-4
GAMMA = 0.995
BATCH_SIZE = 256  # Mega-batches for ultra-aggressive learning
MEMORY_CAPACITY = 50000

# Advanced epsilon strategy with warmup
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995  # Slower decay for better exploration
EPSILON_WARMUP_STEPS = 1000

# Mega-Enhanced Training Configuration
DQN_TRAINING_INTERVAL = 1   # Train after every single experience for maximum learning
DQN_TARGET_UPDATE_INTERVAL = 500  # Reduced frequency for ultra-stability

# Ultra-Advanced Rainbow DQN Features - All cutting-edge techniques enabled
DQN_DOUBLE_DQN = True
DQN_DUELING = True
DQN_PRIORITIZED_REPLAY = True
DQN_SOFT_UPDATE_TAU = 0.001  # Ultra-soft updates for maximum stability
DQN_GRADIENT_CLIP = 0.5      # Tighter gradient clipping

# Mega-Deep Network Architecture (2,000,000+ parameters)
DQN_HIDDEN_LAYERS = [2048, 1024, 512, 256, 128, 64]  # Mega-deep architecture
DQN_USE_BATCH_NORM = False   # Layer norm is better for advanced networks
DQN_USE_LAYER_NORM = True
DQN_DROPOUT_RATE = 0.1       # Reduced dropout for deeper networks
DQN_ACTIVATION = 'SiLU'      # Swish activation (better than ReLU)

# Rainbow DQN Advanced Features
DQN_USE_NOISY_NETWORKS = True     # Automatic exploration
DQN_USE_DISTRIBUTIONAL_RL = True  # C51 distributional learning
DQN_USE_MULTI_STEP = True         # Multi-step learning
DQN_USE_ATTENTION = True          # Self-attention mechanism
DQN_N_STEP = 5                    # Multi-step learning horizon
DQN_NUM_ATOMS = 51               # Distributional RL atoms
DQN_V_MIN = -200                 # Value distribution minimum
DQN_V_MAX = 200                  # Value distribution maximum

# Mega-Advanced Training Parameters
DQN_LEARNING_RATE = 1e-3  # 3x higher learning rate for mega-aggressive learning
DQN_WEIGHT_DECAY = 1e-4
DQN_OPTIMIZER_BETAS = (0.9, 0.999)
DQN_OPTIMIZER_EPS = 1e-4
DQN_USE_COSINE_ANNEALING = True
DQN_SCHEDULER_T0 = 1000
DQN_SCHEDULER_TMULT = 2
DQN_SCHEDULER_ETA_MIN = 1e-6

# Ultra-Prioritized Experience Replay
PER_ALPHA = 0.6              # Priority exponent
PER_BETA = 0.4               # Importance sampling weight
PER_BETA_INCREMENT = 0.001   # Beta annealing rate
PER_MIN_PRIORITY = 1e-6      # Minimum priority

# Advanced Learning Features
DQN_USE_CURRICULUM = True         # Curriculum learning
DQN_CURRICULUM_FACTOR = 1.0       # Curriculum weighting
DQN_DIFFICULTY_THRESHOLD = 0.8    # Difficulty adaptation threshold
DQN_USE_HINDSIGHT = True          # Hindsight Experience Replay

# ===================================================================
# --- Offloading / Accuracy Thresholds ---
# ===================================================================

CPU_UTIL_THRESHOLD_FOR_OFFLOAD = 0.80
QUEUE_LENGTH_THRESHOLD_FOR_OFFLOAD = 10

# ===================================================================
# --- Experiment / Output Configuration ---
# ===================================================================

EXPERIMENT_STRATEGIES = ['dqn', 'drl', 'always_edge', 'always_cloud', 'random']

# ===================================================================
# --- DRL (Deep Reinforcement Learning) Parameters ---
# ===================================================================

# Actor-Critic specific parameters
DRL_LEARNING_RATE = 3e-4
DRL_ENTROPY_COEFFICIENT = 0.01
DRL_VALUE_LOSS_COEFFICIENT = 0.5
DRL_COORDINATION_INTERVAL = 10

# Policy gradient parameters
DRL_ADVANTAGE_NORMALIZATION = True
DRL_GAE_LAMBDA = 0.95
DRL_GRADIENT_CLIP = 0.5

RESULTS_DIR = 'results'
CHART_OUTPUT_DIR = f'{RESULTS_DIR}/charts'
DATA_OUTPUT_DIR = f'{RESULTS_DIR}/data'
CHART_STYLE = 'seaborn-v0_8-darkgrid'

PERFORMANCE_REPORT_FILENAME = 'performance_summary_report.txt'
PERFORMANCE_DATA_FILENAME = 'detailed_performance_metrics.json'
