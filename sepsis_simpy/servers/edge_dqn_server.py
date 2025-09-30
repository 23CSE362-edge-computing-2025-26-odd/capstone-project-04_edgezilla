from dqn_system.edge_dqn_agent import EdgeDQNAgent

class EdgeDQNAgentServer:
    """
    A simulated server to manage and interact with EdgeDQN Agents.
    This replaces a real Flask server for simulation purposes.
    """
    def __init__(self, num_edge_servers, state_size, action_size):
        self.agents = [
            EdgeDQNAgent(state_size, action_size, agent_id=i)
            for i in range(num_edge_servers)
        ]
        print(f"Initialized {num_edge_servers} Edge DQN agents.")

    def choose_action(self, agent_id, state):
        """Simulates the '/choose_action' endpoint."""
        if agent_id < len(self.agents):
            return self.agents[agent_id].choose_action(state)
        return None

    def store_transition(self, agent_id, experience):
        """Simulates the '/store_transition' endpoint."""
        if agent_id < len(self.agents):
            state, action, reward, next_state, done = experience
            self.agents[agent_id].store_experience(state, action, reward, next_state, done)
            return {"status": "success"}
        return {"status": "error", "message": "Agent not found"}

    def trigger_learning(self, agent_id):
        """Triggers a learning step for the specified agent."""
        if agent_id < len(self.agents):
            self.agents[agent_id].update_model()
            return {"status": "learning triggered"}
        return {"status": "error", "message": "Agent not found"}

    def health_check(self):
        """Simulates the '/health' endpoint."""
        return {"status": "ok", "num_agents": len(self.agents)}