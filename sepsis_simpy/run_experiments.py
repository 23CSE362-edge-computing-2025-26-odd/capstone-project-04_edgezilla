# run_experiments.py (Enhanced for Full Reporting with ML Server Integration)

import time
import os
import requests
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

def check_ml_server_status():
    """Check if ML server is running and accessible."""
    try:
        response = requests.get("http://localhost:5002/health", timeout=3)
        if response.status_code == 200:
            server_info = response.json()
            print(f"ML Server Status: {server_info.get('status', 'unknown')}")
            print(f"   Model Trained: {server_info.get('model_trained', False)}")
            return True
    except requests.exceptions.RequestException:
        print("ML Server: Not accessible at http://localhost:5002")
        print("   Using fallback heuristic detection")
        return False
    return False

def validate_csv_data():
    """Validate that CSV data is available and properly formatted."""
    try:
        generator = HealthDataGenerator()
        if generator.data is not None:
            print(f"CSV Data: {len(generator.data)} records loaded from dataset")
            
            # Check for required columns
            required_cols = ['heart_rate', 'blood_oxygen', 'temperature', 'movement']
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
    
    print("Sepsis Detection System Simulation with ML Integration")
    print("=" * 60)
    print(f"Selected strategies: {', '.join(args.strategies)}")
    print(f"Simulation duration: {args.duration} seconds")
    print(f"Results directory: {args.results_dir}")
    
    # Pre-experiment setup and validation
    print("\nSystem Validation:")
    ml_available = check_ml_server_status()
    csv_available = validate_csv_data()
    
    if not ml_available:
        print("\nTo enable ML server:")
        print("   1. Run: python ml_server.py")
        print("   2. Server will be available at http://localhost:5002")
        print("   3. Rerun experiments for full ML integration")
    
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
    print(f"ML Integration: {'Active' if ml_available else 'Fallback Mode'}")
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
