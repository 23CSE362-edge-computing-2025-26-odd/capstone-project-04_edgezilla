# run_experiments.py (Enhanced for Full Reporting)

import time
import os
import config
from main_simulator import SepsisSimulation
from analysis.visualizer import LearningVisualizer, PerformanceVisualizer
from utils.logger import setup_logging

# Import the new, dedicated exporter
from analysis.metrics_exporter import MetricsExporter
# The old exporter can be removed or deprecated
# from analysis.results_exporter import DataExporter

def run_single_experiment(strategy):
    """Runs the simulation and generates all performance reports and charts."""
    print(f"\n{'='*20} RUNNING EXPERIMENT: {strategy.upper()} {'='*20}")
    
    # 1. Run the simulation
    # The 'run' method now returns the final, aggregated metrics dictionary
    simulation = SepsisSimulation(strategy=strategy)
    final_metrics = simulation.run()
    
    # 2. Export the new, comprehensive reports
    exporter = MetricsExporter(final_metrics, strategy_name=strategy)
    exporter.export_to_json()
    exporter.export_to_summary_report()

    # 3. Generate Visualizations (This part is mostly unchanged)
    # Note: For detailed charts, you might want to log data points during the
    # simulation and convert them to DataFrames, similar to Iteration 3's
    # metrics_collector.py. For this prompt, we focus on the final aggregated reports.
    print(f"Visualizations for '{strategy}' would be generated here.")
    print(f"{'='*20} FINISHED EXPERIMENT: {strategy.upper()} {'='*20}")

def main():
    setup_logging(log_file="simulation_run.log")
    print("Starting Comprehensive Sepsis Detection System Simulation...")
    start_time = time.time()

    for strategy in config.EXPERIMENT_STRATEGIES:
        run_single_experiment(strategy)

    end_time = time.time()
    print(f"\nAll experiments completed in {end_time - start_time:.2f} seconds.")
    print(f"All performance reports saved to the '{config.RESULTS_DIR}/' directory.")

if __name__ == "__main__":
    main()
