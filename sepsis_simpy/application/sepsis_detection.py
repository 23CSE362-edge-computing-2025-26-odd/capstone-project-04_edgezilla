import logging

class SepsisApplicationModules:
    """Container for all logical modules of the sepsis detection application."""
    def __init__(self, dqn_server, state_tracker):
        # The decision module needs access to the DQN and state tracker
        self.offload_decision_module = self.OffloadDecisionModule(dqn_server, state_tracker)
        self.logger = logging.getLogger(self.__class__.__name__)

    class PatientClientModule:
        def __init__(self):
            self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__qualname__}")
            
        def process(self, data):
            # Basic processing, perhaps formatting data
            self.logger.info(f"PatientClientModule processed data for Wearable-{data['device_id']}")
            return data

    class PreprocessorModule:
        def __init__(self):
            self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__qualname__}")
            
        def process(self, data):
            # Data cleaning, feature extraction
            self.logger.info("PreprocessorModule cleaned data.")
            data['features_extracted'] = True
            return data

    class OffloadDecisionModule:
        def __init__(self, dqn_server, state_tracker):
            self.dqn_server = dqn_server
            self.state_tracker = state_tracker
            self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__qualname__}")

        def decide(self, data):
            """Calls the DQN to make a routing decision."""
            edge_id = data['ward_id']
            # 1. Get current system state
            current_state = self.state_tracker.get_edge_state(edge_id)
            # 2. Choose action (0=edge, 1=cloud)
            action = self.dqn_server.choose_action(edge_id, current_state)
            self.logger.info(f"OffloadDecisionModule: State={current_state.round(2)}, Action={['PROCESS_ON_EDGE', 'OFFLOAD_TO_CLOUD'][action]}")
            return action, current_state

    class InferenceEdgeModule:
        def __init__(self):
            self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__qualname__}")
            
        def process(self, data):
            # Simulate local sepsis detection
            self.logger.info("InferenceEdgeModule running local detection.")
            data['prediction'] = 'low_risk'
            return data

    class InferenceCloudModule:
        def __init__(self):
            self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__qualname__}")
            
        def process(self, data):
            # Simulate complex analytics on the cloud
            self.logger.info("InferenceCloudModule running complex analytics.")
            data['prediction'] = 'high_risk_sepsis_correlated'
            return data