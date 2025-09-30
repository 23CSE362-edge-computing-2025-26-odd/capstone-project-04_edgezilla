from main_simulator import SepsisSimulation
from utils.logger import setup_logging
import logging
import time

def main():
    """Main function to run the simulation."""
    # Setup logging to write to simulation_run.log
    setup_logging(log_file="simulation_run.log")
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing Sepsis Detection CFRL System Simulation (Iteration 1)...")
    start_time = time.time()

    # Create and run the simulation
    simulation = SepsisSimulation()
    dataframes = simulation.run()

    end_time = time.time()
    logger.info(f"Total real-world execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()