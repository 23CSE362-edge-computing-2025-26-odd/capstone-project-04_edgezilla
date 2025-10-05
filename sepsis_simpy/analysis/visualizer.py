import matplotlib.pyplot as plt
import seaborn as sns
import os

class LearningVisualizer:
    """Creates charts related to DQN learning progress."""
    def __init__(self, df, output_dir='results/charts'):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_reward_progression(self, window=100):
        """Plots the moving average of rewards over time."""
        plt.figure(figsize=(12, 6))
        self.df['reward_ma'] = self.df['reward'].rolling(window=window).mean()
        sns.lineplot(data=self.df, x='timestamp', y='reward_ma', hue='agent_id', palette='viridis')
        plt.title(f'Reward Progression (Moving Average, Window={window})')
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'reward_progression.png'))
        plt.close()

    def plot_epsilon_decay(self):
        """Plots the decay of epsilon over time."""
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=self.df, x='timestamp', y='epsilon', hue='agent_id')
        plt.title('Epsilon Decay Over Time')
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Epsilon (Exploration Rate)')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'epsilon_decay.png'))
        plt.close()

class PerformanceVisualizer:
    """Creates charts related to overall system performance."""
    def __init__(self, perf_df, decision_df, output_dir='results/charts'):
        self.perf_df = perf_df
        self.decision_df = decision_df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_latency_comparison(self, strategy_name=''):
        """Compares latency distributions for edge vs. cloud processing."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.perf_df, x='processing_location', y='latency')
        plt.title(f'Task Latency by Processing Location ({strategy_name})')
        plt.xlabel('Processing Location')
        plt.ylabel('End-to-End Latency (s)')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(self.output_dir, f'latency_comparison_{strategy_name}.png'))
        plt.close()

    def plot_action_distribution(self, strategy_name=''):
        """Plots the distribution of offloading actions."""
        plt.figure(figsize=(8, 5))
        self.decision_df['action'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title(f'Offloading Decision Distribution ({strategy_name})')
        plt.ylabel('') # Hide the y-label
        plt.savefig(os.path.join(self.output_dir, f'action_distribution_{strategy_name}.png'))
        plt.close()