# run_experiments.py (Enhanced for Full Reporting)

import time
import os
import config
from main_simulator import SepsisSimulation
from utils.logger import setup_logging
from analysis.enhanced_metrics_exporter import EnhancedMetricsExporter
from analysis.enhanced_visualizer import EnhancedVisualizer
from analysis.metrics_adapter import MetricsAdapter

def run_single_experiment(strategy):
    """Runs the simulation and generates all performance reports and charts."""
    print(f"\n{'='*20} RUNNING EXPERIMENT: {strategy.upper()} {'='*20}")
    
    # 1. Run the simulation
    # The 'run' method now returns the final, aggregated metrics dictionary
    simulation = SepsisSimulation(strategy=strategy)
    final_metrics = simulation.run()
    
    # 2. Convert metrics to the required format
    converted_metrics = MetricsAdapter.convert_metrics(final_metrics, strategy)
    
    # 3. Export comprehensive metrics to CSV files
    metrics_exporter = EnhancedMetricsExporter(converted_metrics, strategy_name=strategy)
    metrics_exporter.export_all()
    print(f"Exported all metrics to CSV files for strategy: {strategy}")

    # 4. Generate enhanced visualizations
    visualizer = EnhancedVisualizer(converted_metrics, strategy_name=strategy)
    visualizer.plot_all()
    print(f"Generated all visualization charts for strategy: {strategy}")
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
