class LocalRewardCalculator:
    """Calculates rewards based on local (edge-level) performance metrics."""
    def calculate_local_reward(self, execution_time, success_status=True, resource_utilization=0.5, action=0, queue_length=0):
        """
        Calculates a sophisticated reward that considers multiple factors:
        - Execution time (lower is better)
        - Success status (failures are heavily penalized)
        - Resource utilization (penalize high edge utilization)
        - Action type (small penalty for cloud offloading)
        - Queue length (penalize long queues at edge)
        
        Args:
            execution_time (float): Time taken to execute the task
            success_status (bool): Whether the task completed successfully
            resource_utilization (float): Current CPU utilization (0-1)
            action (int): 0 for edge processing, 1 for cloud offloading
            queue_length (int): Current length of the processing queue
        """
        # Base reward inversely proportional to execution time
        reward = 2.0 / (execution_time + 0.01)
        
        # Penalize failures heavily
        if not success_status:
            reward -= 20.0
            
        # Edge processing specific penalties (action == 0)
        if action == 0:
            # Penalize high resource utilization with progressive severity
            if resource_utilization > 0.7:
                penalty = (resource_utilization - 0.7) * 3.0
                reward *= max(0.1, 1.0 - penalty)
                
            # Penalize long queue lengths
            if queue_length > 5:
                queue_penalty = min(0.8, (queue_length - 5) * 0.1)
                reward *= (1.0 - queue_penalty)
        
        # Cloud offloading penalty (action == 1)
        else:
            # Small constant penalty for using cloud resources
            cloud_cost_penalty = 0.2
            reward -= cloud_cost_penalty
            
            # But if edge is heavily loaded, reduce the cloud penalty
            if resource_utilization > 0.8 or queue_length > 8:
                reward += cloud_cost_penalty * 0.5  # Partial refund of the penalty
                
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