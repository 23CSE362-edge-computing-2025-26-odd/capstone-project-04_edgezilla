import logging
from .data_generator import PatientStateModel

class SepsisApplicationModules:
    """Container for all logical modules of the sepsis detection application."""
    def __init__(self, dqn_server, state_tracker):
        # Initialize ML-based sepsis detection model
        self.patient_state_model = PatientStateModel()
        
        # The decision module needs access to the DQN and state tracker
        self.offload_decision_module = self.OffloadDecisionModule(dqn_server, state_tracker)
        
        # Initialize inference modules with ML model
        self.inference_edge_module = self.InferenceEdgeModule(self.patient_state_model)
        self.inference_cloud_module = self.InferenceCloudModule(self.patient_state_model)
        
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
        def __init__(self, patient_state_model):
            self.patient_state_model = patient_state_model
            self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__qualname__}")
            
        def process(self, data):
            # Use ML server for sepsis detection
            sepsis_risk = self.patient_state_model.calculate_sepsis_risk(data)
            patient_state = self.patient_state_model.update_patient_state(data)
            
            self.logger.info(f"InferenceEdgeModule: Sepsis risk={sepsis_risk}, Patient state={patient_state}")
            data['sepsis_prediction'] = sepsis_risk
            data['patient_state'] = patient_state
            data['prediction'] = 'sepsis_detected' if sepsis_risk == 1 else 'normal'
            return data

    class InferenceCloudModule:
        def __init__(self, patient_state_model):
            self.patient_state_model = patient_state_model
            self.logger = logging.getLogger(f"{__class__.__module__}.{__class__.__qualname__}")
            
        def process(self, data):
            # Use ML server for sepsis detection with additional cloud analytics
            sepsis_risk = self.patient_state_model.calculate_sepsis_risk(data)
            patient_state = self.patient_state_model.update_patient_state(data)
            
            # Cloud can perform additional analysis
            confidence_score = sepsis_risk * 0.9 + 0.1  # Simulate confidence scoring
            
            self.logger.info(f"InferenceCloudModule: Sepsis risk={sepsis_risk}, Patient state={patient_state}, Confidence={confidence_score:.2f}")
            data['sepsis_prediction'] = sepsis_risk
            data['patient_state'] = patient_state
            data['confidence_score'] = confidence_score
            data['prediction'] = 'sepsis_detected_high_confidence' if sepsis_risk == 1 else 'normal_verified'
            return data