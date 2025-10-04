from main_simulator import SepsisSimulation
from utils.logger import setup_logging
import logging
import time
import argparse
import config

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
        print(f"⚡ Overriding simulation duration: {config.SIMULATION_DURATION}s → {args.duration}s")
        config.SIMULATION_DURATION = args.duration
    
    # Setup logging to write to simulation_run.log
    setup_logging(log_file="simulation_run.log")
    logger = logging.getLogger(__name__)
    
    logger.info(f"Initializing Sepsis Detection CFRL System Simulation (Strategy: {args.strategy.upper()})...")
    print(f"🏥 Running {args.strategy.upper()} simulation for {args.duration} seconds...")
    start_time = time.time()

    # Create and run the simulation with specified strategy
    simulation = SepsisSimulation(strategy=args.strategy)
    dataframes = simulation.run()

    end_time = time.time()
    print(f"✅ Simulation completed in {end_time - start_time:.2f} real seconds")
    logger.info(f"Total real-world execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()