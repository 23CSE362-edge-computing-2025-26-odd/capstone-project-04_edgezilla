# analysis/metrics_exporter.py

import json
import os
import config

class MetricsExporter:
    """Exports final, aggregated performance metrics to files."""
    def __init__(self, converted_metrics, strategy_name, output_dir=None):
        self.converted_metrics = converted_metrics
        self.strategy_name = strategy_name
        default_dir = os.path.join(config.DATA_OUTPUT_DIR, strategy_name)
        target_dir = output_dir if output_dir is not None else default_dir
        self.output_dir = os.fspath(target_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def export_to_json(self):
        """Saves the detailed metrics dictionary to a JSON file."""
        filepath = os.path.join(self.output_dir, config.PERFORMANCE_DATA_FILENAME)
        # Convert DataFrames to dict for JSON serialization
        json_data = {}
        for key, df in self.converted_metrics.items():
            json_data[key] = df.to_dict('records')
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=4, default=str)
        print(f"Detailed performance data saved to {filepath}")

    def export_to_summary_report(self):
        """Generates a human-readable text file summarizing key metrics."""
        filepath = os.path.join(self.output_dir, config.PERFORMANCE_REPORT_FILENAME)
        with open(filepath, 'w') as f:
            f.write(f"--- Performance Summary for Strategy: {self.strategy_name.upper()} ---\n\n")
            
            # Get performance DataFrame
            perf_df = self.converted_metrics.get('performance')
            if perf_df is None or perf_df.empty:
                f.write("No performance data available.\n")
                return
                
            # Convert latency to milliseconds for display
            perf_df = perf_df.copy()
            perf_df['latency_ms'] = perf_df['latency'] * 1000

            # Latency
            f.write("## Latency Metrics ##\n")
            
            # Overall latency statistics
            overall_mean_ms = perf_df['latency_ms'].mean()
            overall_median_ms = perf_df['latency_ms'].median()
            overall_p95_ms = perf_df['latency_ms'].quantile(0.95)
            
            f.write(f"- Overall Latency:\n")
            f.write(f"    - Average: {self._format_latency(overall_mean_ms)}\n")
            f.write(f"    - Median:  {self._format_latency(overall_median_ms)}\n")
            f.write(f"    - 95th Percentile: {self._format_latency(overall_p95_ms)}\n\n")
            
            # Latency by location
            for location in perf_df['location'].unique():
                location_data = perf_df[perf_df['location'] == location]['latency_ms']
                if len(location_data) > 0:
                    mean_ms = location_data.mean()
                    median_ms = location_data.median()
                    p95_ms = location_data.quantile(0.95)
                    
                    f.write(f"- {location.title()} Latency:\n")
                    f.write(f"    - Average: {self._format_latency(mean_ms)}\n")
                    f.write(f"    - Median:  {self._format_latency(median_ms)}\n")
                    f.write(f"    - 95th Percentile: {self._format_latency(p95_ms)}\n")
                    f.write(f"    - Task Count: {len(location_data)}\n\n")
            # Throughput
            f.write("## Throughput Metrics ##\n")
            total_tasks = len(perf_df)
            # Calculate simulation duration from timestamps
            if len(perf_df) > 1:
                sim_duration = (perf_df['timestamp'].max() - perf_df['timestamp'].min()).total_seconds()
                throughput = total_tasks / sim_duration if sim_duration > 0 else 0
            else:
                throughput = 0
            f.write(f"- Total Tasks Processed: {total_tasks}\n")
            f.write(f"- Overall Throughput: {throughput:.2f} tasks/sec\n\n")

            # ML Inference metrics
            ml_metrics = self.converted_metrics.get('ml_inference')
            if ml_metrics:
                f.write("## ML Inference Metrics ##\n")
                f.write(f"- ML Inference Latency:\n")
                f.write(f"    - Average: {self._format_latency(ml_metrics['mean_ms'])}\n")
                f.write(f"    - Median:  {self._format_latency(ml_metrics['median_ms'])}\n")
                f.write(f"    - 95th Percentile: {self._format_latency(ml_metrics['p95_ms'])}\n")
                f.write(f"    - 99th Percentile: {self._format_latency(ml_metrics['p99_ms'])}\n")
                f.write(f"    - Min: {self._format_latency(ml_metrics['min_ms'])}\n")
                f.write(f"    - Max: {self._format_latency(ml_metrics['max_ms'])}\n")
                f.write(f"    - Std Dev: {self._format_latency(ml_metrics['std_ms'])}\n")
                f.write(f"- Total ML Inferences: {ml_metrics['count']}\n\n")

            # Energy (if available in decisions dataframe or calculated separately)
            decisions_df = self.converted_metrics.get('decisions')
            if decisions_df is not None and not decisions_df.empty:
                f.write("## Resource Utilization ##\n")
                avg_cpu_util = decisions_df['cpu_util'].mean() * 100
                f.write(f"- Average CPU Utilization: {avg_cpu_util:.1f}%\n")
                
                # Count edge vs cloud decisions
                edge_decisions = len(decisions_df[decisions_df['action'] == 'edge'])
                cloud_decisions = len(decisions_df[decisions_df['action'] == 'cloud'])
                f.write(f"- Edge Processing: {edge_decisions} tasks ({edge_decisions/total_tasks:.1%})\n")
                f.write(f"- Cloud Processing: {cloud_decisions} tasks ({cloud_decisions/total_tasks:.1%})\n\n")
            
            # DQN Learning metrics (if available)
            learning_df = self.converted_metrics.get('learning')
            if learning_df is not None and not learning_df.empty:
                f.write("## DQN Learning Metrics ##\n")
                avg_reward = learning_df['reward'].mean()
                final_epsilon = learning_df['epsilon'].iloc[-1]
                avg_loss = learning_df['loss'].mean()
                f.write(f"- Average Reward: {avg_reward:.4f}\n")
                f.write(f"- Final Epsilon: {final_epsilon:.4f}\n")
                f.write(f"- Average Loss: {avg_loss:.4f}\n\n")
            else:
                f.write("## DQN Decision Accuracy ##\n")
                f.write("- Not applicable for this strategy.\n\n")

        print(f"Human-readable performance report saved to {filepath}")
        
    def _format_latency(self, value_ms):
        """Format latency values with appropriate precision based on magnitude"""
        if value_ms < 0.01:  # < 0.01ms, use scientific notation
            return f"{value_ms:.2e} ms"
        elif value_ms < 0.1:  # < 0.1ms, show 4 decimal places
            return f"{value_ms:.4f} ms"
        elif value_ms < 1.0:  # < 1ms, show 3 decimal places
            return f"{value_ms:.3f} ms"
        else:  # >= 1ms, show 2 decimal places
            return f"{value_ms:.2f} ms"
