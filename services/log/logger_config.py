import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure logging system with file and console handlers.
    
    Args:
        log_dir: Directory to store log files
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(levelname)-8s | %(message)s"
    )
    
    # File handler - detailed log (all levels)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"pipeline_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - simplified (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Get configured logger instance."""
    return logging.getLogger("pipeline")
