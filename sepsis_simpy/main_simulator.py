import simpy
import random
import numpy as np
import logging
import config
from infrastructure.devices import WearableDevice, EdgeServer, CloudDataCenter
from infrastructure.network import Network
from dqn_system.state_tracker import SystemStateTracker
from dqn_system.reward_calculator import LocalRewardCalculator
from application.workflow import DataFlowManager
from application.sepsis_detection import SepsisApplicationModules
from servers.server_manager import ServerManager
from analysis.metrics_collector import SimulationMetrics

class SepsisSimulation:
    def __init__(self, strategy='dqn'):
        self.env = simpy.Environment()
        self.strategy = strategy
        self.metrics = SimulationMetrics()
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- START OF CORRECTION ---

        # Initialize all components
        self.cloud = CloudDataCenter(self.env)
        self.edge_servers = [EdgeServer(self.env, i) for i in range(config.NUM_WARDS)]
        self.wearable_devices = self._create_wearables()
        self.network = Network(self.env)
        
        self.infrastructure = {
            'edge_servers': self.edge_servers,
            'cloud': self.cloud
        }

        # Initialize DQN-specific objects to None first
        self.server_manager = None
        self.state_tracker = None
        self.reward_calc = None

        # Setup DQN components ONLY if the strategy is 'dqn'
        if self.strategy == 'dqn':
            self.server_manager = ServerManager(state_size=5, action_size=2)
            self.server_manager.start_servers()
            self.state_tracker = SystemStateTracker(self.infrastructure)
            self.reward_calc = LocalRewardCalculator()
        
        # This part is now safe for all strategies
        self.app_modules = SepsisApplicationModules(
            dqn_server=getattr(self.server_manager, 'edge_server', None),
            state_tracker=self.state_tracker
        )
        self.workflow_manager = DataFlowManager(self.env, self.app_modules, self.infrastructure, self.network)


    def _create_wearables(self):
        """Helper to create and link wearable devices."""
        devices = []
        for i in range(config.TOTAL_WEARABLES):
            ward_id = i // config.WEARABLES_PER_WARD
            edge_server = self.edge_servers[ward_id]
            devices.append(WearableDevice(self.env, i, ward_id, edge_server))
        return devices

    def run_patient_monitoring(self, wearable):
        """Main process for a single wearable device, now handles learning."""
        last_state = self.state_tracker.get_edge_state(wearable.ward_id) if self.strategy == 'dqn' else None

        while True:
            health_data = yield self.env.process(wearable.generate_health_data())
            health_data['ward_id'] = wearable.ward_id
            
            # Use the correct offloading strategy
            if self.strategy == 'dqn':
                action, current_state = self.app_modules.offload_decision_module.decide(health_data)
            elif self.strategy == 'always_edge':
                action = 0
            elif self.strategy == 'always_cloud':
                action = 1
            elif self.strategy == 'random':
                action = random.choice([0, 1])
            
            # For non-DQN strategies, state is just for recording
            if self.strategy != 'dqn':
                 current_state = np.zeros(5) # Dummy state
                 
            self.metrics.record_decision(self.env.now, wearable.ward_id, action, current_state)

            # Manually manage the workflow based on the chosen action
            result = yield self.env.process(self._execute_task(health_data, action))
            
            self.metrics.record_latency(self.env.now, wearable.ward_id, result['execution_time'], action)

            # --- Learning Step for DQN ---
            if self.strategy == 'dqn':
                reward = self.reward_calc.calculate_local_reward(result['execution_time'])
                next_state = self.state_tracker.get_edge_state(wearable.ward_id)
                done = self.env.now >= config.SIMULATION_DURATION
                
                # Store experience
                experience = (last_state, action, reward, next_state, done)
                self.server_manager.edge_server.store_transition(wearable.ward_id, experience)
                
                # Trigger learning and update target network periodically
                if self.env.now % config.DQN_TRAINING_INTERVAL < config.SENSOR_DATA_GENERATION_INTERVAL:
                    self.server_manager.edge_server.trigger_learning(wearable.ward_id)
                if self.env.now % config.DQN_TARGET_UPDATE_INTERVAL < config.SENSOR_DATA_GENERATION_INTERVAL:
                     self.server_manager.edge_server.agents[wearable.ward_id].update_target_network()

                self.metrics.update_learning_metrics(
                    self.env.now, wearable.ward_id, reward, 0, # Loss is internal to agent
                    self.server_manager.edge_server.agents[wearable.ward_id].epsilon
                )
                last_state = next_state # Update state for next transition
            
            yield self.env.timeout(config.SENSOR_DATA_GENERATION_INTERVAL)
            
    def _execute_task(self, data, action):
        """Simplified execution based on action."""
        start_time = self.env.now
        if action == 0: # Edge
            edge = self.edge_servers[data['ward_id']]
            proc_time = config.TASK_CPU_REQUIREMENT / edge.cpu_capacity
            with edge.cpu.request() as req:
                yield req
                yield self.env.timeout(proc_time)
        else: # Cloud
            yield self.env.timeout(config.LATENCY_EDGE_TO_CLOUD / 1000.0)
            proc_time = (config.TASK_CPU_REQUIREMENT * 2) / self.cloud.cpu_capacity
            with self.cloud.resource_pools.request() as req:
                yield req
                yield self.env.timeout(proc_time)
        
        return {"execution_time": self.env.now - start_time}

    def run(self):
        """Initializes and runs the simulation."""
        self.logger.info(f"Starting Simulation for Strategy: {self.strategy.upper()}")
        for wearable in self.wearable_devices:
            self.env.process(self.run_patient_monitoring(wearable))
        self.env.run(until=config.SIMULATION_DURATION)
        self.logger.info(f"Simulation Finished for Strategy: {self.strategy.upper()}")
        return self.metrics.get_dataframes()