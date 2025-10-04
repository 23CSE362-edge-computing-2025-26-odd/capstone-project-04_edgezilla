# analysis/metrics_exporter.py

import json
import os
import config

class MetricsExporter:
    """Exports final, aggregated performance metrics to files."""
    def __init__(self, metrics, strategy_name):
        self.metrics = metrics
        self.strategy_name = strategy_name
        self.output_dir = os.path.join(config.DATA_OUTPUT_DIR, strategy_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def export_to_json(self):
        """Saves the detailed metrics dictionary to a JSON file."""
        filepath = os.path.join(self.output_dir, config.PERFORMANCE_DATA_FILENAME)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Detailed performance data saved to {filepath}")

    def export_to_summary_report(self):
        """Generates a human-readable text file summarizing key metrics."""
        filepath = os.path.join(self.output_dir, config.PERFORMANCE_REPORT_FILENAME)
        with open(filepath, 'w') as f:
            f.write(f"--- Performance Summary for Strategy: {self.strategy_name.upper()} ---\n\n")

            # Latency
            f.write("## Latency Metrics ##\n")
            for name, data in self.metrics.get('latency', {}).items():
                f.write(f"- {name.replace('_', ' ').title()}:\n")
                
                # Convert to milliseconds and format based on magnitude
                mean_ms = data['mean_s'] * 1000
                median_ms = data['median_s'] * 1000
                p95_ms = data['p95_s'] * 1000
                
                def format_latency(value_ms):
                    """Format latency values with appropriate precision based on magnitude"""
                    if value_ms < 0.01:  # < 0.01ms, use scientific notation
                        return f"{value_ms:.2e} ms"
                    elif value_ms < 0.1:  # < 0.1ms, show 4 decimal places
                        return f"{value_ms:.4f} ms"
                    elif value_ms < 1.0:  # < 1ms, show 3 decimal places
                        return f"{value_ms:.3f} ms"
                    else:  # >= 1ms, show 2 decimal places
                        return f"{value_ms:.2f} ms"
                
                f.write(f"    - Average: {format_latency(mean_ms)}\n")
                f.write(f"    - Median:  {format_latency(median_ms)}\n")
                f.write(f"    - 95th Percentile: {format_latency(p95_ms)}\n\n")

            # Throughput
            f.write("## Throughput Metrics ##\n")
            throughput = self.metrics.get('throughput', {})
            total_tasks = throughput.get('completed_tasks', {}).get('task_processed', 0)
            f.write(f"- Total Tasks Processed: {total_tasks}\n")
            f.write(f"- Overall Throughput: {throughput.get('overall_tasks_per_second', 0):.2f} tasks/sec\n\n")

            # Energy
            f.write("## Energy Efficiency ##\n")
            energy = self.metrics.get('energy', {})
            total_energy = energy.get('total_system_energy_joules', 0)
            f.write(f"- Total System Energy: {total_energy:.2f} Joules\n")
            if total_tasks > 0:
                f.write(f"- Energy per Task: {total_energy / total_tasks:.2f} Joules\n\n")
            f.write("Component Energy Breakdown:\n")
            for comp, e_val in energy.get('components', {}).items():
                 f.write(f"    - {comp}: {e_val:.2f} J ({e_val/total_energy:.1%})\n")
            f.write("\n")

            # Accuracy
            f.write("## DQN Decision Accuracy ##\n")
            accuracy = self.metrics.get('accuracy', {})
            if accuracy.get("accuracy_rate", "N/A") != "N/A":
                f.write(f"- Accuracy vs Optimal Policy: {accuracy['accuracy_rate'] * 100:.2f}%\n")
                f.write(f"- Total Decisions Analyzed: {accuracy['total_decisions']}\n\n")
            else:
                f.write("- Not applicable for this strategy.\n\n")

        print(f"Human-readable performance report saved to {filepath}")
