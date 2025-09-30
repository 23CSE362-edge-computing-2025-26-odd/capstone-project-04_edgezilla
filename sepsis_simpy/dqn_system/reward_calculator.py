class LocalRewardCalculator:
    """Calculates rewards based on local (edge-level) performance metrics."""
    def calculate_local_reward(self, execution_time, success_status=True, resource_utilization=0.5):
        """
        Calculates a reward for a single action.
        A lower execution time should result in a higher reward.
        """
        # The core component of the reward is inverse of execution time
        # We add a small constant to avoid division by zero
        reward = 1.0 / (execution_time + 0.01)

        # Penalize for failures (if applicable)
        if not success_status:
            reward -= 10.0 # Heavy penalty for failure

        # Optional: Penalize for very high resource utilization to encourage load balancing
        if resource_utilization > 0.9:
            reward *= (1.0 - (resource_utilization - 0.9))
            
        return reward

class GlobalRewardCalculator:
    """Calculates rewards based on system-wide performance."""
    def calculate_global_reward(self, fairness, throughput, resource_efficiency):
        """
        Combines multiple system-wide objectives into a single reward signal.
        This is a placeholder for more complex logic.
        """
        # Weighted sum of normalized metrics
        reward = (0.4 * fairness) + (0.4 * throughput) + (0.2 * resource_efficiency)
        return reward

    def distribute_to_edges(self, global_reward, num_edges):
        """Distributes the global reward among the edge agents."""
        # Simple even distribution for now
        return [global_reward / num_edges] * num_edges