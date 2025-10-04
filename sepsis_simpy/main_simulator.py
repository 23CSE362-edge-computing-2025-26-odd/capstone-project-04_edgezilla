# main_simulator.py (Corrected for Accurate Monitoring)

import simpy
import random
import numpy as np
import config
from infrastructure.devices import WearableDevice, EdgeServer, CloudDataCenter
from infrastructure.network import Network
from dqn_system.state_tracker import SystemStateTracker
from dqn_system.reward_calculator import LocalRewardCalculator
from servers.server_manager import ServerManager
from analysis.performance_monitor import PerformanceMonitor
from application.data_generator import PatientStateModel

class SepsisSimulation:
    def __init__(self, strategy='dqn'):
        self.env = simpy.Environment()
        self.strategy = strategy
        self.monitor = PerformanceMonitor() # Create the monitor instance

        # --- Infrastructure setup ---
        self.cloud = CloudDataCenter(self.env)
        self.edge_servers = [EdgeServer(self.env, i) for i in range(config.NUM_WARDS)]
        self.wearable_devices = self._create_wearables()
        self.network = Network(self.env)
        self.infrastructure = {'edge_servers': self.edge_servers, 'cloud': self.cloud}

        # --- Logic for handling different strategies ---
        self.server_manager = None
        self.state_tracker = None
        # Initialize ML-based sepsis detection model
        self.patient_state_model = PatientStateModel()
        
        if self.strategy == 'dqn':
            self.server_manager = ServerManager(state_size=5, action_size=2)
            self.server_manager.start_servers()
            self.state_tracker = SystemStateTracker(self.infrastructure)

    def _create_wearables(self):
        """Helper to create and link wearable devices."""
        devices = []
        for i in range(config.TOTAL_WEARABLES):
            ward_id = i // config.WEARABLES_PER_WARD
            devices.append(WearableDevice(self.env, i, ward_id, self.edge_servers[ward_id]))
        return devices

    def run_patient_monitoring(self, wearable):
        """Main process, now using env.now for simulation latencies."""
        last_state = self.state_tracker.get_edge_state(wearable.ward_id) if self.strategy == 'dqn' else None

        while True:
            # --- START: Correct End-to-End Latency Measurement ---
            e2e_start_time = self.env.now

            health_data = yield self.env.process(wearable.generate_health_data())
            health_data['ward_id'] = wearable.ward_id
            self.monitor.energy.record_network_energy('wireless', config.HEALTH_DATA_PACKET_SIZE_KB)

            # Perform sepsis detection using ML server
            sepsis_prediction = self.patient_state_model.calculate_sepsis_risk(health_data)
            patient_state = self.patient_state_model.update_patient_state(health_data)
            health_data['sepsis_prediction'] = sepsis_prediction
            health_data['patient_state'] = patient_state
            
            print(f"Time {self.env.now:.2f}: Wearable-{wearable.device_id} - Sepsis Risk: {sepsis_prediction}, State: {patient_state}")

            action, current_state = self._make_offloading_decision(health_data)
            if self.strategy == 'dqn':
                self.monitor.accuracy.evaluate_decision(current_state, action)
            
            processing_duration = yield self.env.process(self._execute_task(health_data, action))
            
            # Log sepsis detection results
            if health_data.get('sepsis_prediction', 0) == 1:
                location = 'edge' if action == 0 else 'cloud'
                print(f"⚠️  SEPSIS ALERT: Patient {wearable.device_id} (Ward {wearable.ward_id}) - Processed on {location.upper()}")

            # --- END: Correct End-to-End Latency Measurement ---
            e2e_latency = self.env.now - e2e_start_time
            self.monitor.sim_latencies['end_to_end_latency'].append(e2e_latency)
            self.monitor.throughput.record_completion('task_processed')
            
            if self.strategy == 'dqn':
                reward_calc = LocalRewardCalculator()
                reward = reward_calc.calculate_local_reward(e2e_latency) # Reward is based on total latency
                next_state = self.state_tracker.get_edge_state(wearable.ward_id)
                done = self.env.now >= config.SIMULATION_DURATION
                experience = (last_state, action, reward, next_state, done)
                self.server_manager.edge_server.store_transition(wearable.ward_id, experience)
                last_state = next_state
            
            yield self.env.timeout(config.SENSOR_DATA_GENERATION_INTERVAL)

    def _make_offloading_decision(self, data):
        """Uses high-precision timer for DQN inference."""
        if self.strategy == 'dqn':
            current_state = self.state_tracker.get_edge_state(data['ward_id'])
            # Use the real-time, high-precision latency tracker
            self.monitor.realtime_latency.start('dqn_inference_latency', data['device_id'])
            action = self.server_manager.edge_server.choose_action(data['ward_id'], current_state)
            self.monitor.realtime_latency.end('dqn_inference_latency', data['device_id'])
            return action, current_state
        
        # Non-DQN strategies
        current_state = np.zeros(5)
        action = 0
        if self.strategy == 'always_cloud': action = 1
        elif self.strategy == 'random': action = random.choice([0, 1])
        return action, current_state

    def _execute_task(self, data, action):
        """Uses env.now to correctly measure simulation queue times."""
        if action == 0: # Process on Edge
            edge = self.edge_servers[data['ward_id']]
            proc_time = config.TASK_CPU_REQUIREMENT / edge.cpu_capacity
            
            queue_arrival_time = self.env.now
            with edge.cpu.request() as req:
                yield req
                # Correctly measure queue wait time
                queue_wait = self.env.now - queue_arrival_time
                self.monitor.sim_latencies['edge_queue_wait'].append(queue_wait)
                
                yield self.env.timeout(proc_time)
            
            processing_duration = self.env.now - queue_arrival_time
            # Energy is based on processing time only, not queue time
            self.monitor.energy.record_device_energy('edge', 'busy', proc_time) 
            return processing_duration
        
        else: # Offload to Cloud
            # Simulate network delay
            yield self.env.timeout(config.LATENCY_EDGE_TO_CLOUD / 1000.0)
            self.monitor.energy.record_network_energy('wired', config.HEALTH_DATA_PACKET_SIZE_KB)

            cloud = self.cloud
            proc_time = (config.TASK_CPU_REQUIREMENT * 2) / cloud.cpu_capacity
            
            queue_arrival_time = self.env.now
            with cloud.resource_pools.request() as req:
                yield req
                # Correctly measure queue wait time (after network delay)
                queue_wait = self.env.now - queue_arrival_time
                self.monitor.sim_latencies['cloud_queue_wait'].append(queue_wait)
                
                yield self.env.timeout(proc_time)

            processing_duration = self.env.now - queue_arrival_time
            self.monitor.energy.record_device_energy('cloud', 'busy', proc_time)
            return processing_duration

    def run(self):
        """Runs the simulation and returns final aggregated metrics."""
        print(f"--- Starting Simulation for Strategy: {self.strategy.upper()} ---")
        for wearable in self.wearable_devices:
            self.env.process(self.run_patient_monitoring(wearable))
        
        self.env.run(until=config.SIMULATION_DURATION)
        
        print(f"--- Simulation Finished for Strategy: {self.strategy.upper()} ---")
        return self.monitor.get_aggregated_results(sim_duration=self.env.now)