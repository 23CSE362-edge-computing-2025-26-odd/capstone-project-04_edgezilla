import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

class EnhancedVisualizer:
    """Creates enhanced charts for simulation analysis."""
    
    def __init__(self, metrics, strategy_name):
        """
        Initialize the visualizer with metrics data and strategy name.
        
        Args:
            metrics: Dictionary containing metrics DataFrames
            strategy_name: Name of the strategy used (e.g., 'dqn', 'always_edge')
        """
        self.metrics = metrics
        self.strategy_name = strategy_name
        self.output_dir = os.path.join('results/charts', strategy_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_reward_distribution(self):
        """Creates a detailed visualization of reward distribution over time."""
        if 'learning' not in self.metrics:
            return
            
        learning_df = self.metrics['learning']
        plt.figure(figsize=(12, 8))
        
        # Create main reward distribution plot
        plt.subplot(2, 1, 1)
        sns.histplot(data=learning_df, x='reward', bins=50, kde=True)
        plt.title(f'Reward Distribution ({self.strategy_name})')
        plt.xlabel('Reward Value')
        plt.ylabel('Count')
        
        # Create reward over time plot
        plt.subplot(2, 1, 2)
        sns.lineplot(data=learning_df, x='timestamp', y='reward', alpha=0.3, color='gray', label='Raw')
        sns.lineplot(data=learning_df, x='timestamp', y='rolling_reward', 
                    color='blue', label='Moving Average (100)')
        plt.title('Reward Progression Over Time')
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reward_distribution.png'), dpi=300)
        plt.close()
        
    def plot_latency_comparison(self):
        """Creates an enhanced latency comparison visualization."""
        perf_df = self.metrics['performance']
        
        plt.figure(figsize=(15, 8))
        
        # Create main latency comparison plot
        plt.subplot(2, 1, 1)
        sns.boxplot(data=perf_df, x='location', y='latency_ms')
        plt.title(f'Task Latency by Processing Location ({self.strategy_name})')
        plt.xlabel('Processing Location')
        plt.ylabel('Latency (ms)')
        
        # Create latency over time plot
        plt.subplot(2, 1, 2)
        sns.lineplot(data=perf_df, x='timestamp', y='latency_ms', 
                    hue='location', alpha=0.3)
        sns.lineplot(data=perf_df, x='timestamp', y='rolling_avg_latency',
                    color='black', label='Overall Moving Average')
        plt.title('Latency Progression Over Time')
        plt.xlabel('Time')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_comparison.png'), dpi=300)
        plt.close()
        
    def plot_epsilon_decay(self):
        """Creates a detailed visualization of epsilon decay."""
        if 'learning' not in self.metrics:
            return
            
        learning_df = self.metrics['learning']
        plt.figure(figsize=(12, 6))
        
        # Plot epsilon decay
        sns.lineplot(data=learning_df, x='timestamp', y='epsilon')
        
        # Add exploration/exploitation regions
        plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, 
                   label='High Exploitation Threshold')
        plt.fill_between(learning_df['timestamp'], 0, 0.2, 
                        color='green', alpha=0.1, label='Exploitation Zone')
        plt.fill_between(learning_df['timestamp'], 0.2, 1, 
                        color='yellow', alpha=0.1, label='Exploration Zone')
        
        plt.title(f'Epsilon Decay Over Time ({self.strategy_name})')
        plt.xlabel('Time')
        plt.ylabel('Epsilon (Exploration Rate)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'epsilon_decay.png'), dpi=300)
        plt.close()
        
    def plot_all(self):
        """Generate all visualization charts."""
        self.plot_reward_distribution()
        self.plot_latency_comparison()
        self.plot_epsilon_decay()