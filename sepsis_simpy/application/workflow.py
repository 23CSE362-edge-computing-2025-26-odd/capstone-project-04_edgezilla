import config

class DataFlowManager:
    """Routes a data tuple through the application modules based on DQN decisions."""
    def __init__(self, env, app_modules, infrastructure, network):
        self.env = env
        self.app = app_modules
        self.infra = infrastructure
        self.net = network

    def handle_data_packet(self, data):
        """
        A SimPy process that defines the workflow for a single data packet.
        """
        # --- Start of Workflow ---
        start_time = self.env.now
        
        # 1. Basic processing on wearable (assumed to be done before sending)
        packet = self.app.PatientClientModule().process(data)
        packet = self.app.PreprocessorModule().process(packet)

        # 2. Make offloading decision
        action, current_state = self.app.offload_decision_module.decide(packet)
        
        # --- Branching based on decision ---
        if action == 0: # Process on Edge
            edge_server = self.infra['edge_servers'][packet['ward_id']]
            processing_time = config.TASK_CPU_REQUIREMENT / edge_server.cpu_capacity
            
            with edge_server.cpu.request() as req:
                yield req
                yield self.env.timeout(processing_time)
                
            self.app.InferenceEdgeModule().process(packet)
            
        else: # action == 1: Offload to Cloud
            # Simulate network latency to cloud
            yield self.env.process(self.net.send_data('edge', 'cloud', packet))
            
            cloud = self.infra['cloud']
            # Simulate more complex task on faster CPU
            processing_time = (config.TASK_CPU_REQUIREMENT * 3) / cloud.cpu_capacity
            
            with cloud.resource_pools.request() as req:
                yield req
                yield self.env.timeout(processing_time)

            self.app.InferenceCloudModule().process(packet)
            
        # --- End of Workflow ---
        end_time = self.env.now
        execution_time = end_time - start_time
        
        # This return value is crucial for calculating rewards and storing experience
        return {
            "execution_time": execution_time,
            "decision_state": current_state,
            "action_taken": action,
            "ward_id": packet['ward_id']
        }