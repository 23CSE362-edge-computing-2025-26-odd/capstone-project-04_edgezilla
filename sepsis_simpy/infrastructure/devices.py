import simpy
import numpy as np
from application.data_generator import HealthDataGenerator
import config

# Shared data generator instance for all wearables
_shared_data_generator = None

class WearableDevice:
    """Represents a patient-monitoring wearable device."""
    def __init__(self, env, device_id, ward_id, edge_server):
        self.env = env
        self.device_id = device_id
        self.ward_id = ward_id
        self.parent_edge = edge_server
        self.cpu_capacity = config.WEARABLE_CPU
        self.memory = config.WEARABLE_RAM
        self.power_consumption = (config.POWER_WEARABLE_IDLE, config.POWER_WEARABLE_BUSY)
        # Use shared data generator instance to avoid row replication
        global _shared_data_generator
        if _shared_data_generator is None:
            _shared_data_generator = HealthDataGenerator()
        self.data_generator = _shared_data_generator

    def generate_health_data(self):
        """Generates a new set of health data."""
        # Simulate sensor reading delay
        yield self.env.timeout(config.LATENCY_WEARABLE_SENSOR / 1000.0)
        data = self.data_generator.generate_all()
        data['device_id'] = self.device_id
        data['timestamp'] = self.env.now
        print(f"Time {self.env.now:.2f}: Wearable-{self.device_id} (Ward-{self.ward_id}) generated data: HR={data['heart_rate']} SpO2={data['blood_oxygen']}%")
        return data

    def get_current_load(self):
        # Placeholder for future iterations
        return np.random.uniform(0.1, 0.5)

class EdgeServer:
    """Represents a ward-level edge server for local processing."""
    def __init__(self, env, server_id):
        self.env = env
        self.server_id = server_id
        self.cpu_capacity = config.EDGE_SERVER_CPU
        self.memory = config.EDGE_SERVER_RAM
        self.connected_wearables = []
        # Use a SimPy Resource to model the CPU with a capacity of 1 (representing 100% utilization)
        self.cpu = simpy.Resource(env, capacity=1)
        self.queue = simpy.Store(env)

    def process_data(self, data):
        """Simulates processing data on the edge server."""
        arrival_time = self.env.now
        print(f"Time {self.env.now:.2f}: Edge-{self.server_id} received data from Wearable-{data['device_id']}.")

        # Calculate processing time based on task size and CPU capacity
        processing_time = config.TASK_CPU_REQUIREMENT / self.cpu_capacity # in seconds

        with self.cpu.request() as request:
            yield request
            print(f"Time {self.env.now:.2f}: Edge-{self.server_id} started processing for Wearable-{data['device_id']}.")
            yield self.env.timeout(processing_time)
            print(f"Time {self.env.now:.2f}: Edge-{self.server_id} finished processing. Total time: {(self.env.now - arrival_time):.2f}s")

    def get_utilization(self):
        """Returns the current CPU utilization of the server."""
        return self.cpu.count / self.cpu.capacity

class CloudDataCenter:
    """Represents the central hospital cloud data center."""
    def __init__(self, env):
        self.env = env
        self.cpu_capacity = config.CLOUD_DATACENTER_CPU
        self.memory = config.CLOUD_DATACENTER_RAM
        self.resource_pools = simpy.Resource(env, capacity=10) # Model as 10 parallel processing units

    def process_complex_analytics(self, data):
        """Placeholder for complex analytics processing in the cloud."""
        print(f"Time {self.env.now:.2f}: Cloud received data for complex analytics.")
        # Simulate more intensive processing
        processing_time = (config.TASK_CPU_REQUIREMENT * 5) / self.cpu_capacity
        with self.resource_pools.request() as request:
            yield request
            yield self.env.timeout(processing_time)
            print(f"Time {self.env.now:.2f}: Cloud finished complex analytics.")