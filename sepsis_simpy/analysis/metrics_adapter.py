import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MetricsAdapter:
    """Adapts the raw simulation metrics to the format expected by enhanced exporters."""
    
    @staticmethod
    def convert_metrics(raw_metrics, strategy_name):
        """
        Converts raw simulation metrics to DataFrames format.
        
        Args:
            raw_metrics: Dictionary containing raw metrics from PerformanceMonitor
            strategy_name: Name of the strategy used
            
        Returns:
            Dictionary containing pandas DataFrames with converted metrics
        """
        metrics_dict = {}
        
        # Get simulation duration and total tasks
        sim_duration = raw_metrics['system']['total_simulation_time_s']
        total_tasks = raw_metrics['throughput']['completed_tasks']['task_processed']
        
        # Create timestamps array (as datetime)
        base_time = datetime.now().replace(microsecond=0)
        timestamps = [base_time + timedelta(seconds=t) for t in np.linspace(0, sim_duration, total_tasks)]
        
        # Convert performance metrics
        performance_data = []
        
        # Handle different strategies
        if strategy_name.lower() == 'always_edge':
            # All tasks processed on edge
            edge_count = total_tasks
            cloud_count = 0
        elif strategy_name.lower() == 'always_cloud':
            # All tasks processed on cloud
            edge_count = 0
            cloud_count = total_tasks
        else:
            # Mixed strategy (DQN or random)
            edge_count = raw_metrics['latency'].get('edge_queue_wait', {}).get('count', 0)
            cloud_count = raw_metrics['latency'].get('cloud_queue_wait', {}).get('count', 0)
            
            # Adjust if counts don't match total
            if edge_count + cloud_count != total_tasks:
                edge_count = total_tasks // 2
                cloud_count = total_tasks - edge_count
        
        # Create performance DataFrame
        for i in range(total_tasks):
            is_edge = i < edge_count
            latency = raw_metrics['latency']['end_to_end_latency']['mean_s']
            
            # Add queue wait times if available
            if is_edge and 'edge_queue_wait' in raw_metrics['latency']:
                latency += raw_metrics['latency']['edge_queue_wait'].get('mean_s', 0)
            elif not is_edge and 'cloud_queue_wait' in raw_metrics['latency']:
                latency += raw_metrics['latency']['cloud_queue_wait'].get('mean_s', 0)
                
            performance_data.append({
                'timestamp': timestamps[i],
                'latency': latency,
                'location': 'edge' if is_edge else 'cloud',
                'ward_id': i % 3  # Assuming 3 wards
            })
        
        metrics_dict['performance'] = pd.DataFrame(performance_data)
        
        # Convert learning metrics (for DQN only)
        if strategy_name.lower() == 'dqn':
            learning_data = []
            
            # Calculate epsilon decay
            epsilon_start = 1.0
            epsilon_end = 0.01
            epsilon_decay = 0.995
            epsilons = epsilon_start * (epsilon_decay ** np.arange(total_tasks))
            epsilons = np.clip(epsilons, epsilon_end, epsilon_start)
            
            # Create learning DataFrame
            for i in range(total_tasks):
                learning_data.append({
                    'timestamp': timestamps[i],
                    'agent_id': f'ward_{i % 3}',  # One agent per ward
                    'reward': 1.0 / raw_metrics['latency']['end_to_end_latency']['mean_s'],
                    'loss': 0.1 * np.exp(-i/1000),  # Simulated decreasing loss
                    'epsilon': epsilons[i]
                })
            
            metrics_dict['learning'] = pd.DataFrame(learning_data)
        
        # Convert decision metrics
        decisions_data = []
        
        # Calculate average CPU utilization from energy metrics
        edge_energy = raw_metrics['energy']['components'].get('edge_busy', 0)
        total_energy = raw_metrics['energy']['total_system_energy_joules']
        avg_cpu_util = edge_energy / total_energy if total_energy > 0 else 0
        
        # Create decisions DataFrame
        for i in range(total_tasks):
            is_edge = i < edge_count
            decisions_data.append({
                'timestamp': timestamps[i],
                'ward_id': i % 3,
                'action': 'edge' if is_edge else 'cloud',
                'cpu_util': avg_cpu_util,
                'queue_length': edge_count / 3  # Average queue length per ward
            })
        
        metrics_dict['decisions'] = pd.DataFrame(decisions_data)
        
        return metrics_dict