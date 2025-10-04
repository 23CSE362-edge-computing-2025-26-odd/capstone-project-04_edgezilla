import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

class EnhancedMetricsExporter:
    """Enhanced metrics exporter that generates detailed CSV files and summary reports."""
    
    def __init__(self, metrics, strategy_name):
        """
        Initialize the exporter with metrics data and strategy name.
        
        Args:
            metrics: Dictionary containing metrics DataFrames
            strategy_name: Name of the strategy used (e.g., 'dqn', 'always_edge')
        """
        self.metrics = metrics
        self.strategy_name = strategy_name
        self.output_dir = os.path.join('results/data', strategy_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_summary_report(self):
        """Generate a comprehensive summary report in CSV format."""
        # Calculate summary statistics
        perf_df = self.metrics['performance']
        decisions_df = self.metrics['decisions']
        learning_df = self.metrics.get('learning', pd.DataFrame())  # May not exist for non-DQN strategies
        
        summary_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': self.strategy_name,
            'total_tasks': len(perf_df),
            'edge_tasks': len(perf_df[perf_df['location'] == 'edge']),
            'cloud_tasks': len(perf_df[perf_df['location'] == 'cloud']),
            'avg_latency_ms': perf_df['latency'].mean() * 1000,
            'p95_latency_ms': perf_df['latency'].quantile(0.95) * 1000,
            'edge_ratio': len(perf_df[perf_df['location'] == 'edge']) / len(perf_df),
            'cloud_ratio': len(perf_df[perf_df['location'] == 'cloud']) / len(perf_df)
        }
        
        if not learning_df.empty:
            summary_data.update({
                'avg_reward': learning_df['reward'].mean(),
                'final_epsilon': learning_df['epsilon'].iloc[-1],
                'avg_loss': learning_df['loss'].mean()
            })
            
        # Create summary DataFrame and export
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(os.path.join(self.output_dir, 'summary_report.csv'), index=False)
        
    def export_performance_metrics(self):
        """Export detailed performance metrics to CSV."""
        perf_df = self.metrics['performance']
        
        # Add derived metrics
        perf_df['latency_ms'] = perf_df['latency'] * 1000
        
        # Calculate rolling averages
        perf_df['rolling_avg_latency'] = perf_df['latency'].rolling(window=50).mean()
        
        # Export to CSV
        perf_df.to_csv(os.path.join(self.output_dir, 'performance.csv'), index=False)
        
    def export_learning_metrics(self):
        """Export DQN learning metrics to CSV."""
        if 'learning' in self.metrics:
            learning_df = self.metrics['learning']
            
            # Add derived metrics
            learning_df['rolling_reward'] = learning_df['reward'].rolling(window=100).mean()
            learning_df['rolling_loss'] = learning_df['loss'].rolling(window=100).mean()
            
            # Export to CSV
            learning_df.to_csv(os.path.join(self.output_dir, 'learning.csv'), index=False)
            
    def export_decision_metrics(self):
        """Export decision-making metrics to CSV."""
        decisions_df = self.metrics['decisions']
        
        # Set timestamp as index for proper resampling
        decisions_df = decisions_df.set_index('timestamp')
        
        # Calculate decision statistics per time window (1-minute)
        window_stats = decisions_df.resample('1Min').agg({
            'action': ['count', lambda x: (x == 'edge').mean()],
            'cpu_util': 'mean',
            'queue_length': 'mean'
        }).reset_index()
        
        # Flatten column names
        window_stats.columns = ['timestamp', 'decisions_count', 'edge_ratio', 'avg_cpu_util', 'avg_queue_len']
        
        # Export both detailed and aggregated data
        decisions_df.reset_index().to_csv(os.path.join(self.output_dir, 'decisions.csv'), index=False)
        window_stats.to_csv(os.path.join(self.output_dir, 'decisions_window_stats.csv'), index=False)
        
    def export_all(self):
        """Export all metrics to their respective CSV files."""
        self.export_summary_report()
        self.export_performance_metrics()
        self.export_learning_metrics()
        self.export_decision_metrics()