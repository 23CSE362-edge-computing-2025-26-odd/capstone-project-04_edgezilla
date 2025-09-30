from .edge_dqn_server import EdgeDQNAgentServer
from .cloud_dqn_server import CloudDQNAgentServer
import config

class ServerManager:
    """Manages the instantiation of the simulated DQN servers."""
    def __init__(self, state_size, action_size):
        self.edge_server = None
        self.cloud_server = None
        self.state_size = state_size
        self.action_size = action_size

    def start_servers(self):
        """Instantiates the server objects."""
        print("ServerManager: Starting DQN servers...")
        self.edge_server = EdgeDQNAgentServer(
            num_edge_servers=config.NUM_WARDS,
            state_size=self.state_size,
            action_size=self.action_size
        )
        self.cloud_server = CloudDQNAgentServer()
        print("ServerManager: All servers are running.")

    def stop_servers(self):
        """Placeholder for any server cleanup logic."""
        print("ServerManager: Stopping servers.")
        self.edge_server = None
        self.cloud_server = None

    def check_server_health(self):
        """Checks the health of the simulated servers."""
        edge_health = self.edge_server.health_check() if self.edge_server else {"status": "down"}
        cloud_health = self.cloud_server.health_check() if self.cloud_server else {"status": "down"}
        print(f"Edge Server Health: {edge_health}")
        print(f"Cloud Server Health: {cloud_health}")
        return edge_health['status'] == 'ok' and cloud_health['status'] == 'ok'