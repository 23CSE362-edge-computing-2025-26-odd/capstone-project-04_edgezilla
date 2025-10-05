# run_experiments.py (Enhanced for Full Reporting with Local ML Inference)

import time
import os
import pandas as pd
import argparse
import config
from main_simulator import SepsisSimulation
from utils.logger import setup_logging
from analysis.enhanced_metrics_exporter import EnhancedMetricsExporter
from analysis.enhanced_visualizer import EnhancedVisualizer
from analysis.metrics_adapter import MetricsAdapter
from analysis.metrics_exporter import MetricsExporter
from application.data_generator import HealthDataGenerator, PatientStateModel



def validate_csv_data():
    """Validate that CSV data is available and properly formatted."""
    try:
        generator = HealthDataGenerator()
        if generator.data is not None:
            print(f"CSV Data: {len(generator.data)} records loaded from dataset")
            
            # Check for required columns (8 parameters for ML model)
            required_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
            missing_cols = [col for col in required_cols if col not in generator.data.columns]
            
            if missing_cols:
                print(f"CSV Warning: Missing columns {missing_cols}")
                return False
            else:
                print("   All required columns present")
                return True
        else:
            print("CSV Data: Not found, using synthetic data generation")
            return False
    except Exception as e:
        print(f"CSV Data Error: {e}")
        return False


def run_single_experiment(strategy):
    """Runs the simulation and generates all performance reports and charts."""
    print(f"\n{'='*20} RUNNING EXPERIMENT: {strategy.upper()} {'='*20}")
    
    print(f"\nStarting simulation with:")
    print(f"   Strategy: {strategy.upper()}")
    print(f"   Duration: {config.SIMULATION_DURATION} seconds")
    
    # 1. Run the simulation with ML integration
    simulation = SepsisSimulation(strategy=strategy)
    final_metrics = simulation.run()
    
    # Log ML-related statistics
    print(f"\nSimulation Results Summary:")
    print(f"   Tasks Processed: {final_metrics['throughput']['completed_tasks']['task_processed']}")
    print(f"   Average Latency: {final_metrics['latency']['end_to_end_latency']['mean_s']:.4f}s")
    print(f"   Total Energy: {final_metrics['energy']['total_system_energy_joules']:.2f}J")
    
    # 2. Convert metrics to the required format
    converted_metrics = MetricsAdapter.convert_metrics(final_metrics, strategy)
    
    # 3. Export comprehensive metrics to CSV files
    metrics_exporter = EnhancedMetricsExporter(converted_metrics, strategy_name=strategy)
    metrics_exporter.export_all()
    print(f"Exported all metrics to CSV files for strategy: {strategy}")
    
    # 4. Generate human-readable summary report
    summary_exporter = MetricsExporter(converted_metrics, strategy)
    summary_exporter.export_to_summary_report()
    print(f"Generated readable performance summary for strategy: {strategy}")

    # 5. Generate enhanced visualizations
    visualizer = EnhancedVisualizer(converted_metrics, strategy_name=strategy)
    visualizer.plot_all()
    print(f"Generated all visualization charts for strategy: {strategy}")
    
    # 6. Display latency breakdown showing ML inference integration
    performance_csv = f"results/data/{strategy}/performance.csv"
    if os.path.exists(performance_csv):
        import pandas as pd
        df = pd.read_csv(performance_csv)
        
        avg_processing = df['latency_ms'].mean()
        avg_ml_inference = df['ml_inference_time_ms'].mean()
        avg_total = df['total_latency_ms'].mean()
        
        print(f"\n{'='*50}")
        print(f"LATENCY BREAKDOWN - {strategy.upper()}")
        print(f"{'='*50}")
        print(f"Processing Latency:     {avg_processing:.2f} ms")
        print(f"ML Inference Time:    + {avg_ml_inference:.2f} ms")
        print(f"{'_'*35}")
        print(f"Total Latency:        = {avg_total:.2f} ms")
        print(f"\nTasks Processed: {len(df)}")
        print(f"Total Energy: {final_metrics['energy']['total_system_energy_joules']:.2f}J")
        print(f"{'='*50}")
    
    print(f"{'='*20} FINISHED EXPERIMENT: {strategy.upper()} {'='*20}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Sepsis Detection Simulation Experiments')
    parser.add_argument('--strategies', nargs='+', default=config.EXPERIMENT_STRATEGIES,
                       choices=['dqn', 'always_edge', 'always_cloud', 'random'],
                       help='Strategies to test (default: all strategies)')
    parser.add_argument('--duration', type=int, default=config.SIMULATION_DURATION,
                       help=f'Simulation duration in seconds (default: {config.SIMULATION_DURATION})')
    parser.add_argument('--results-dir', type=str, default=config.RESULTS_DIR,
                       help=f'Directory to save results (default: {config.RESULTS_DIR})')
    args = parser.parse_args()
    
    # Override config values if specified
    if args.duration != config.SIMULATION_DURATION:
        print(f"Overriding simulation duration: {config.SIMULATION_DURATION}s → {args.duration}s")
        config.SIMULATION_DURATION = args.duration
    
    if args.results_dir != config.RESULTS_DIR:
        print(f"Overriding results directory: {config.RESULTS_DIR} → {args.results_dir}")
        config.RESULTS_DIR = args.results_dir
        # Ensure the directory exists
        os.makedirs(args.results_dir, exist_ok=True)
    
    setup_logging(log_file="simulation_run.log")
    
    print("Sepsis Detection System Simulation with Local ML Inference")
    print("=" * 60)
    print(f"Selected strategies: {', '.join(args.strategies)}")
    print(f"Simulation duration: {args.duration} seconds")
    print(f"Results directory: {args.results_dir}")
    
    # Pre-experiment setup and validation
    print("\nSystem Validation:")
    csv_available = validate_csv_data()
    
    print(f"\nRunning {len(args.strategies)} strategies: {', '.join(args.strategies)}")
    print(f"Results will be saved to: {args.results_dir}/")
    
    start_time = time.time()
    
    # Run all experiments
    for i, strategy in enumerate(args.strategies, 1):
        print(f"\n[{i}/{len(args.strategies)}] Executing strategy: {strategy}")
        run_single_experiment(strategy)
    
    end_time = time.time()
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE COMPLETED")
    print("=" * 60)
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")
    print(f"Strategies Tested: {len(args.strategies)}")
    print(f"Results Location: {args.results_dir}/")
    print(f"Data Source: {'CSV Dataset' if csv_available else 'Synthetic Generation'}")
    print(f"Simulation Duration: {args.duration} seconds per strategy")
    
    print("\nGenerated Files:")
    print("   • Performance metrics (CSV)")
    print("   • Decision logs (CSV)")
    print("   • Learning metrics (CSV for DQN)")
    print("   • Latency comparison charts (PNG)")
    print("   • Performance visualizations (PNG)")
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
