import logging
import os
from datetime import datetime
from pathlib import Path

# Global logger instance
_logger = None

def setup_logger(log_dir):
    """Set up logger with both file and console handlers."""
    global _logger
    
    # If logger is already initialized, return it
    if _logger is not None:
        return _logger
    
    if log_dir is None:
        raise ValueError("log_dir must be provided to setup_logger")
        
    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('answering_agent')
    logger.handlers = []  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Store logger instance
    _logger = logger
    return _logger

def get_logger():
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logger first.")
    return _logger 