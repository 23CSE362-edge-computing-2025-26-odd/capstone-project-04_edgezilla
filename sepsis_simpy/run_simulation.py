from main_simulator import SepsisSimulation
from utils.logger import setup_logging
from analysis.metrics_exporter import MetricsExporter
from analysis.metrics_adapter import MetricsAdapter
import logging
import time
import argparse
import config
import os

def main():
    """Main function to run the simulation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Sepsis Detection Simulation')
    parser.add_argument('--strategy', type=str, default='dqn', 
                       choices=['dqn', 'always_edge', 'always_cloud', 'random'],
                       help='Strategy to use for task offloading (default: dqn)')
    parser.add_argument('--duration', type=int, default=config.SIMULATION_DURATION,
                       help=f'Simulation duration in seconds (default: {config.SIMULATION_DURATION})')
    args = parser.parse_args()
    
    # Override config duration if specified
    if args.duration != config.SIMULATION_DURATION:
        print(f"Overriding simulation duration: {config.SIMULATION_DURATION}s → {args.duration}s")
        config.SIMULATION_DURATION = args.duration
    
    # Setup logging to write to simulation_run.log
    setup_logging(log_file="simulation_run.log")
    logger = logging.getLogger(__name__)
    
    logger.info(f"Initializing Sepsis Detection CFRL System Simulation (Strategy: {args.strategy.upper()})...")
    print(f"Running {args.strategy.upper()} simulation for {args.duration} seconds...")
    start_time = time.time()

    # Create and run the simulation with specified strategy
    simulation = SepsisSimulation(strategy=args.strategy)
    final_metrics = simulation.run()

    # Generate complete output suite: CSV data, visualizations, and summary report
    converted_metrics = MetricsAdapter.convert_metrics(final_metrics, args.strategy)
    
    # 1. Export comprehensive metrics to CSV files
    from analysis.enhanced_metrics_exporter import EnhancedMetricsExporter
    enhanced_exporter = EnhancedMetricsExporter(converted_metrics, strategy_name=args.strategy)
    enhanced_exporter.export_all()
    print(f"Exported all metrics to CSV files for strategy: {args.strategy}")
    
    # 2. Generate human-readable summary report
    summary_exporter = MetricsExporter(converted_metrics, args.strategy)
    summary_exporter.export_to_summary_report()
    print(f"Generated readable performance summary for strategy: {args.strategy}")
    
    # 3. Generate enhanced visualizations
    from analysis.enhanced_visualizer import EnhancedVisualizer
    visualizer = EnhancedVisualizer(converted_metrics, strategy_name=args.strategy)
    visualizer.plot_all()
    print(f"Generated all visualization charts for strategy: {args.strategy}")
    
    # 4. Display latency breakdown showing ML inference integration
    performance_csv = f"results/data/{args.strategy}/performance.csv"
    if os.path.exists(performance_csv):
        import pandas as pd
        df = pd.read_csv(performance_csv)
        
        avg_processing = df['latency_ms'].mean()
        avg_ml_inference = df['ml_inference_time_ms'].mean()
        avg_total = df['total_latency_ms'].mean()
        
        print(f"\n{'='*50}")
        print(f"LATENCY BREAKDOWN - {args.strategy.upper()}")
        print(f"{'='*50}")
        print(f"Processing Latency:     {avg_processing:.2f} ms")
        print(f"ML Inference Time:    + {avg_ml_inference:.2f} ms")
        print(f"{'_'*35}")
        print(f"Total Latency:        = {avg_total:.2f} ms")
        print(f"\nTasks Processed: {len(df)}")
        print(f"Total Energy: {final_metrics['energy']['total_system_energy_joules']:.2f}J")
        print(f"{'='*50}")
    else:
        print(f"\nQuick Summary:")
        print(f"   Tasks Processed: {final_metrics['throughput']['completed_tasks']['task_processed']}")
        print(f"   Average Latency: {final_metrics['latency']['end_to_end_latency']['mean_s']*1000:.2f}ms")
        print(f"   Total Energy: {final_metrics['energy']['total_system_energy_joules']:.2f}J")
    print(f"\nAll outputs saved to: results/data/{args.strategy}/ and results/charts/{args.strategy}/")
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} real seconds")
    logger.info(f"Total real-world execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()