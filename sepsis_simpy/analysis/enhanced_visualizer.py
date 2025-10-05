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
        
    def create_ml_inference_chart(self, output_path):
        """Create ML inference time visualization."""
        if 'performance' not in self.metrics or self.metrics['performance'].empty:
            print("No performance data available for ML inference visualization")
            return
        
        perf_df = self.metrics['performance']
        if 'ml_inference_time_ms' not in perf_df.columns:
            print("No ML inference time data available")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create ML inference time distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=perf_df, x='ml_inference_time_ms', bins=30, kde=True)
        plt.title(f'ML Inference Time Distribution ({self.strategy_name})')
        plt.xlabel('ML Inference Time (ms)')
        plt.ylabel('Count')
        
        # ML inference over time
        plt.subplot(2, 2, 2)
        sns.lineplot(data=perf_df, x='timestamp', y='ml_inference_time_ms', alpha=0.7)
        plt.title('ML Inference Time Over Time')
        plt.xlabel('Time')
        plt.ylabel('ML Inference Time (ms)')
        plt.xticks(rotation=45)
        
        # ML inference by location
        plt.subplot(2, 2, 3)
        sns.boxplot(data=perf_df, x='location', y='ml_inference_time_ms')
        plt.title('ML Inference Time by Processing Location')
        plt.xlabel('Processing Location')
        plt.ylabel('ML Inference Time (ms)')
        
        # ML vs total latency correlation
        plt.subplot(2, 2, 4)
        perf_df['total_latency_ms'] = perf_df['latency'] * 1000
        sns.scatterplot(data=perf_df, x='ml_inference_time_ms', y='total_latency_ms', 
                       hue='location', alpha=0.6)
        plt.title('ML Inference vs Total Latency')
        plt.xlabel('ML Inference Time (ms)')
        plt.ylabel('Total Latency (ms)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ML inference chart saved to: {output_path}")

    def plot_all(self):
        """Generate all visualization charts."""
        self.plot_reward_distribution()
        self.plot_latency_comparison()
        self.plot_epsilon_decay()
        self.create_ml_inference_chart(os.path.join(self.output_dir, 'ml_inference_analysis.png'))