import logging
import sys

def setup_logging(log_level=logging.INFO, log_file="simulation.log"):
    """Configures the root logger for the entire application."""
    log_formatter = logging.Formatter(
        '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log each run
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def get_logger(name):
    """Returns a logger instance for a specific module."""
    return logging.getLogger(name)