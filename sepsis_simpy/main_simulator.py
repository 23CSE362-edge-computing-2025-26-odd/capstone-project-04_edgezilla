# main_simulator.py (Corrected for Accurate Monitoring)

import simpy
import random
import numpy as np
import config
from infrastructure.devices import WearableDevice, EdgeServer, CloudDataCenter
from infrastructure.network import Network
from dqn_system.state_tracker import SystemStateTracker
from dqn_system.reward_calculator import LocalRewardCalculator
from drl_system.single_edge_drl_agent import SepsisAwareDRLAgent
from servers.server_manager import ServerManager
from analysis.performance_monitor import PerformanceMonitor, AccuracyEvaluator, SepsisAwareDRLAccuracyEvaluator
from application.data_generator import PatientStateModel

class SepsisSimulation:
    def __init__(self, strategy='dqn'):
        self.env = simpy.Environment()
        self.strategy = strategy
        self.monitor = PerformanceMonitor() # Create the monitor instance
        
        # Initialize accuracy evaluators based on strategy
        if strategy == 'dqn':
            self.monitor.accuracy = AccuracyEvaluator()
        elif strategy == 'drl':
            self.monitor.sepsis_drl_accuracy = SepsisAwareDRLAccuracyEvaluator()

        # --- Infrastructure setup ---
        self.cloud = CloudDataCenter(self.env)
        self.edge_servers = [EdgeServer(self.env, i) for i in range(config.NUM_WARDS)]
        self.wearable_devices = self._create_wearables()
        self.network = Network(self.env)
        self.infrastructure = {'edge_servers': self.edge_servers, 'cloud': self.cloud}

        # --- Logic for handling different strategies ---
        self.server_manager = None
        self.state_tracker = None
        
        # --- Initialize Accuracy Evaluators ---
        self.accuracy_evaluator = None
        self.sepsis_aware_drl_agents = None
        
        # Initialize ML-based sepsis detection models for edge and cloud
        self.edge_ml_model = PatientStateModel(device_type='edge')
        self.cloud_ml_model = PatientStateModel(device_type='cloud')
        
        if self.strategy == 'dqn':
            self.server_manager = ServerManager(state_size=5, action_size=2)
            self.server_manager.start_servers()
            self.state_tracker = SystemStateTracker(self.infrastructure)
            self.accuracy_evaluator = AccuracyEvaluator()
        elif self.strategy == 'drl':
            print("Initializing Sepsis-Aware DRL agents...")
            self.state_tracker = SystemStateTracker(self.infrastructure)
            # Initialize sepsis-aware DRL agents (one per ward, can route to cloud)
            self.sepsis_aware_drl_agents = {}
            for i in range(config.NUM_WARDS):
                self.sepsis_aware_drl_agents[i] = SepsisAwareDRLAgent(
                    state_size=7,  # Enhanced with patient vitals
                    action_size=2, # 0: edge, 1: cloud
                    learning_rate=config.DRL_LEARNING_RATE
                )
            print(f"Initialized {len(self.sepsis_aware_drl_agents)} Sepsis-Aware DRL agents with cloud routing capability.")

    def _create_wearables(self):
        """Helper to create and link wearable devices."""
        devices = []
        for i in range(config.TOTAL_WEARABLES):
            ward_id = i // config.WEARABLES_PER_WARD
            devices.append(WearableDevice(self.env, i, ward_id, self.edge_servers[ward_id]))
        return devices

    def run_patient_monitoring(self, wearable):
        """Main process, now using env.now for simulation latencies."""
        last_state = None
        last_action = None
        last_log_prob = None
        if self.strategy in ['dqn', 'drl']:
            last_state = self.state_tracker.get_edge_state(wearable.ward_id)

        while True:
            # --- START: Correct End-to-End Latency Measurement ---
            e2e_start_time = self.env.now

            health_data = yield self.env.process(wearable.generate_health_data())
            health_data['ward_id'] = wearable.ward_id
            self.monitor.energy.record_network_energy('wireless', config.HEALTH_DATA_PACKET_SIZE_KB)

            action, current_state = self._make_offloading_decision(health_data)
            
            # Update last_action and last_log_prob for DRL learning
            if self.strategy == 'drl':
                last_action = health_data.get('drl_action', action)
                last_log_prob = health_data.get('drl_log_prob', None)
            
            if self.strategy == 'dqn':
                # Pass patient data for medical accuracy evaluation
                patient_data = {
                    'HR': health_data.get('HR', 75),
                    'SpO2': health_data.get('SpO2', 98),
                    'device_id': health_data.get('device_id', 0)
                }
                self.monitor.accuracy.evaluate_decision(current_state, action, patient_data)
            elif self.strategy == 'drl':
                # Evaluate single-edge DRL decision with patient data
                performance_data = {
                    'latency': 0.0,  # Will be updated after processing
                    'cpu_util': current_state[0] if current_state is not None else 0.5,
                    'queue_length': current_state[2] if current_state is not None else 0
                }
                patient_data = {
                    'HR': health_data.get('HR', 75),
                    'SpO2': health_data.get('SpO2', 98),
                    'device_id': health_data.get('device_id', 0)
                }
                self.monitor.sepsis_drl_accuracy.evaluate_decision(
                    current_state, action, performance_data, patient_data
                )
            
            # Execute task with ML inference (sepsis detection happens during processing)
            processing_duration, sepsis_result = yield self.env.process(self._execute_task(health_data, action))
            
            # Extract results from task execution
            sepsis_prediction = sepsis_result['sepsis_prediction']
            patient_state = sepsis_result['patient_state']
            health_data['sepsis_prediction'] = sepsis_prediction
            health_data['patient_state'] = patient_state
            
            print(f"Time {self.env.now:.2f}: Wearable-{wearable.device_id} - Sepsis Risk: {sepsis_prediction}, State: {patient_state}")
            
            # Log sepsis detection results
            if health_data.get('sepsis_prediction', 0) == 1:
                location = 'edge' if action == 0 else 'cloud'
                print(f"SEPSIS ALERT: Patient {wearable.device_id} (Ward {wearable.ward_id}) - Processed on {location.upper()}")

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
            elif self.strategy == 'drl':
                # Single-edge DRL learning (no cloud coordination)
                reward_calc = LocalRewardCalculator()
                
                # Calculate edge-only reward based on local processing efficiency
                cpu_util = current_state[0] if current_state is not None else 0.5
                queue_len = current_state[2] if current_state is not None else 0
                # Get patient data for reward calculation
                patient_data = health_data.get('patient_data', {
                    'HR': health_data.get('HR', 75),
                    'SpO2': health_data.get('SpO2', 98)
                })
                
                sepsis_reward = self.sepsis_aware_drl_agents[wearable.ward_id].calculate_sepsis_aware_reward(
                    e2e_latency, cpu_util, queue_len, action, patient_data
                )
                
                next_state = self.state_tracker.get_edge_state(wearable.ward_id)
                done = self.env.now >= config.SIMULATION_DURATION
                
                # Store experience in sepsis-aware DRL agent
                if last_state is not None and last_log_prob is not None:
                    self.sepsis_aware_drl_agents[wearable.ward_id].store_experience(
                        last_state, last_action, sepsis_reward, next_state, done, last_log_prob, patient_data
                    )
                    
                    # Update policy periodically or at episode end
                    if done or (self.env.now % 20 < config.SENSOR_DATA_GENERATION_INTERVAL):
                        self.sepsis_aware_drl_agents[wearable.ward_id].update_policy(episode_complete=done)
                
                last_state = next_state
            
            yield self.env.timeout(config.SENSOR_DATA_GENERATION_INTERVAL)

    def _make_offloading_decision(self, data):
        """Make offloading decision based on strategy (DQN, DRL, or heuristic)."""
        if self.strategy == 'dqn':
            current_state = self.state_tracker.get_edge_state(data['ward_id'])
            # Use the real-time, high-precision latency tracker
            self.monitor.realtime_latency.start('dqn_inference_latency', data['device_id'])
            action = self.server_manager.edge_server.choose_action(data['ward_id'], current_state)
            self.monitor.realtime_latency.end('dqn_inference_latency', data['device_id'])
            return action, current_state
            
        elif self.strategy == 'drl':
            current_state = self.state_tracker.get_edge_state(data['ward_id'])
            
            # Sepsis-aware DRL decision with patient vitals
            patient_data = {
                'HR': data.get('HR', 75),
                'SpO2': data.get('SpO2', 98),
                'device_id': data.get('device_id', 0)
            }
            
            action, log_prob = self.sepsis_aware_drl_agents[data['ward_id']].select_action(
                current_state, patient_data=patient_data, training=True
            )
            
            # Store action and log_prob for later experience storage
            data['drl_action'] = action
            data['drl_log_prob'] = log_prob
            data['patient_data'] = patient_data  # Store for reward calculation
            
            return action, current_state
        
        # Non-learning strategies (always_edge, always_cloud, random)
        current_state = np.zeros(5)
        action = 0
        if self.strategy == 'always_cloud': 
            action = 1
        elif self.strategy == 'random': 
            action = random.choice([0, 1])
        # 'always_edge' stays as action = 0
        
        return action, current_state

    def _execute_task(self, data, action):
        """Execute task with ML inference on the appropriate device."""
        if action == 0: # Process on Edge
            edge = self.edge_servers[data['ward_id']]
            proc_time = config.TASK_CPU_REQUIREMENT / edge.cpu_capacity
            
            queue_arrival_time = self.env.now
            with edge.cpu.request() as req:
                yield req
                # Correctly measure queue wait time
                queue_wait = self.env.now - queue_arrival_time
                self.monitor.sim_latencies['edge_queue_wait'].append(queue_wait)
                
                # Perform ML inference during edge processing
                sepsis_prediction = self.edge_ml_model.calculate_sepsis_risk(data)
                patient_state = self.edge_ml_model.update_patient_state(data)
                ml_inference_time = self.edge_ml_model.last_inference_time_ms / 1000.0  # Convert to seconds
                
                # Record ML inference latency for edge
                self.monitor.realtime_latency.start('edge_ml_inference_latency', data['device_id'])
                yield self.env.timeout(ml_inference_time)  # Simulate ML inference time
                self.monitor.realtime_latency.end('edge_ml_inference_latency', data['device_id'])
                
                # Continue with regular processing (ML inference time already included above)
                yield self.env.timeout(proc_time)
            
            processing_duration = self.env.now - queue_arrival_time
            self.monitor.energy.record_device_energy('edge', 'busy', proc_time) 
            
            return processing_duration, {
                'sepsis_prediction': sepsis_prediction,
                'patient_state': patient_state
            }
        
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
                
                # Perform ML inference during cloud processing
                sepsis_prediction = self.cloud_ml_model.calculate_sepsis_risk(data)
                patient_state = self.cloud_ml_model.update_patient_state(data)
                ml_inference_time = self.cloud_ml_model.last_inference_time_ms / 1000.0  # Convert to seconds
                
                # Record ML inference latency for cloud
                self.monitor.realtime_latency.start('cloud_ml_inference_latency', data['device_id'])
                yield self.env.timeout(ml_inference_time)  # Simulate ML inference time
                self.monitor.realtime_latency.end('cloud_ml_inference_latency', data['device_id'])
                
                # Continue with regular processing (ML inference time already included above)
                yield self.env.timeout(proc_time)

            processing_duration = self.env.now - queue_arrival_time
            self.monitor.energy.record_device_energy('cloud', 'busy', proc_time)
            
            return processing_duration, {
                'sepsis_prediction': sepsis_prediction,
                'patient_state': patient_state
            }

    def run(self):
        """Runs the simulation and returns final aggregated metrics."""
        print(f"--- Starting Simulation for Strategy: {self.strategy.upper()} ---")
        for wearable in self.wearable_devices:
            self.env.process(self.run_patient_monitoring(wearable))
        
        # Run the simulation
        self.env.run(until=config.SIMULATION_DURATION)
        
        # DRL episode cleanup (sepsis-aware agents)
        if self.strategy == 'drl':
            # Final policy updates for sepsis-aware DRL agents
            for agent in self.sepsis_aware_drl_agents.values():
                agent.update_policy(episode_complete=True)
        
        print(f"--- Simulation Finished for Strategy: {self.strategy.upper()} ---")
        return self.monitor.get_aggregated_results(sim_duration=self.env.now)