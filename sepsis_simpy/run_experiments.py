import time
import os
import config
from main_simulator import SepsisSimulation
from analysis.results_exporter import DataExporter
from analysis.visualizer import LearningVisualizer, PerformanceVisualizer
from utils.logger import setup_logging
import logging

def run_single_experiment(strategy):
    """Runs the simulation for a single strategy and saves its results."""
    logger = logging.getLogger(__name__)
    logger.info(f"{'='*20} RUNNING EXPERIMENT: {strategy.upper()} {'='*20}")

    simulation = SepsisSimulation(strategy=strategy)
    dataframes = simulation.run()
    
    strategy_charts_dir = os.path.join(config.CHART_OUTPUT_DIR, strategy)
    strategy_data_dir = os.path.join(config.DATA_OUTPUT_DIR, strategy)

    exporter = DataExporter(dataframes, output_dir=strategy_data_dir)
    exporter.export_to_csv(strategy_name=strategy)
    exporter.generate_summary_report(strategy_name=strategy)

    perf_df = dataframes.get('performance')
    decision_df = dataframes.get('decisions')
    
    if perf_df is not None and not perf_df.empty:
        perf_viz = PerformanceVisualizer(perf_df, decision_df, output_dir=strategy_charts_dir)
        perf_viz.plot_latency_comparison(strategy_name=strategy)
        perf_viz.plot_action_distribution(strategy_name=strategy)

    if strategy == 'dqn':
        learning_df = dataframes.get('learning')
        if learning_df is not None and not learning_df.empty:
            learn_viz = LearningVisualizer(learning_df, output_dir=strategy_charts_dir)
            learn_viz.plot_reward_progression()
            learn_viz.plot_epsilon_decay()
            
    logger.info(f"{'='*20} FINISHED EXPERIMENT: {strategy.upper()} {'='*20}")


def main():
    """Main function to set up and run all experiments."""
    setup_logging(log_file="simulation_run.log")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Comprehensive Sepsis Detection System Simulation...")
    start_time = time.time()
    for strategy in config.EXPERIMENT_STRATEGIES:
        run_single_experiment(strategy)

    end_time = time.time()
    logger.info(f"All experiments completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"All results have been saved to the '{config.RESULTS_DIR}/' directory.")


if __name__ == "__main__":
    main()