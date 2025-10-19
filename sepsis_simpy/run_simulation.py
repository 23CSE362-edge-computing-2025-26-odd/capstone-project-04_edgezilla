from main_simulator import SepsisSimulation
from utils.logger import setup_logging
from analysis.metrics_exporter import MetricsExporter
from analysis.metrics_adapter import MetricsAdapter
import logging
import time
import argparse
import config
import os


def run_simulation(strategy='dqn', duration=None, num_wards=None, patients_per_ward=None,
                   data_output_dir=None, charts_output_dir=None):
    """Run a simulation with optional workload and output overrides."""
    requested_duration = duration if duration is not None else config.SIMULATION_DURATION
    original_settings = {
        'SIMULATION_DURATION': config.SIMULATION_DURATION,
        'NUM_WARDS': config.NUM_WARDS,
        'WEARABLES_PER_WARD': config.WEARABLES_PER_WARD,
        'TOTAL_WEARABLES': config.TOTAL_WEARABLES,
    }

    setup_logging(log_file="simulation_run.log")
    logger = logging.getLogger(__name__)

    if requested_duration != config.SIMULATION_DURATION:
        print(f"Overriding simulation duration: {config.SIMULATION_DURATION}s → {requested_duration}s")
    if num_wards is not None or patients_per_ward is not None:
        print("Applying workload override for simulation")

    config.SIMULATION_DURATION = requested_duration
    if num_wards is not None:
        config.NUM_WARDS = num_wards
    if patients_per_ward is not None:
        config.WEARABLES_PER_WARD = patients_per_ward
    config.TOTAL_WEARABLES = config.NUM_WARDS * config.WEARABLES_PER_WARD

    logger.info(
        "Initializing Sepsis Detection CFRL System Simulation (Strategy: %s, Wards: %s, Patients/Ward: %s)",
        strategy.upper(), config.NUM_WARDS, config.WEARABLES_PER_WARD
    )
    print(
        f"Running {strategy.upper()} simulation | duration={requested_duration}s | wards={config.NUM_WARDS} | "
        f"patients_per_ward={config.WEARABLES_PER_WARD}"
    )
    start_time = time.time()

    try:
        simulation = SepsisSimulation(strategy=strategy)
        final_metrics = simulation.run()

        converted_metrics = MetricsAdapter.convert_metrics(final_metrics, strategy)

        from analysis.enhanced_metrics_exporter import EnhancedMetricsExporter
        enhanced_exporter = EnhancedMetricsExporter(
            converted_metrics,
            strategy_name=strategy,
            output_dir=data_output_dir
        )
        enhanced_exporter.export_all()
        print(f"Exported all metrics to CSV files for strategy: {strategy}")

        summary_exporter = MetricsExporter(
            converted_metrics,
            strategy,
            output_dir=enhanced_exporter.output_dir
        )
        summary_exporter.export_to_summary_report()
        print(f"Generated readable performance summary for strategy: {strategy}")

        from analysis.enhanced_visualizer import EnhancedVisualizer
        visualizer = EnhancedVisualizer(
            converted_metrics,
            strategy_name=strategy,
            output_dir=charts_output_dir
        )
        visualizer.plot_all()
        print(f"Generated all visualization charts for strategy: {strategy}")

        performance_csv = os.path.join(enhanced_exporter.output_dir, 'performance.csv')
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
        else:
            print(f"\nQuick Summary:")
            print(f"   Tasks Processed: {final_metrics['throughput']['completed_tasks']['task_processed']}")
            print(f"   Average Latency: {final_metrics['latency']['end_to_end_latency']['mean_s']*1000:.2f}ms")
            print(f"   Total Energy: {final_metrics['energy']['total_system_energy_joules']:.2f}J")

        end_time = time.time()
        logger.info("Total real-world execution time: %.2f seconds.", end_time - start_time)
        print(f"\nAll outputs saved to: {enhanced_exporter.output_dir}/ and {visualizer.output_dir}/")
        print(f"Simulation completed in {end_time - start_time:.2f} real seconds")

        return {
            'final_metrics': final_metrics,
            'converted_metrics': converted_metrics,
            'data_dir': enhanced_exporter.output_dir,
            'charts_dir': visualizer.output_dir,
            'duration': requested_duration,
            'num_wards': config.NUM_WARDS,
            'patients_per_ward': config.WEARABLES_PER_WARD,
        }
    finally:
        config.SIMULATION_DURATION = original_settings['SIMULATION_DURATION']
        config.NUM_WARDS = original_settings['NUM_WARDS']
        config.WEARABLES_PER_WARD = original_settings['WEARABLES_PER_WARD']
        config.TOTAL_WEARABLES = original_settings['TOTAL_WEARABLES']


def main():
    """Main function to run the simulation with CLI overrides."""
    parser = argparse.ArgumentParser(description='Run Sepsis Detection Simulation')
    parser.add_argument('--strategy', type=str, default='dqn',
                        choices=['dqn', 'drl', 'always_edge', 'always_cloud', 'random'],
                        help='Strategy to use for task offloading (default: dqn)')
    parser.add_argument('--duration', type=int, default=config.SIMULATION_DURATION,
                        help=f'Simulation duration in seconds (default: {config.SIMULATION_DURATION})')
    parser.add_argument('--num-wards', type=int, help='Override number of hospital wards for this run')
    parser.add_argument('--patients-per-ward', type=int,
                        help='Override number of patients per ward (avg wearable count)')
    parser.add_argument('--data-output-dir', type=str,
                        help='Directory to write exported CSV metrics')
    parser.add_argument('--charts-output-dir', type=str,
                        help='Directory to write generated charts')
    args = parser.parse_args()

    run_simulation(
        strategy=args.strategy,
        duration=args.duration,
        num_wards=args.num_wards,
        patients_per_ward=args.patients_per_ward,
        data_output_dir=args.data_output_dir,
        charts_output_dir=args.charts_output_dir
    )


if __name__ == "__main__":
    main()