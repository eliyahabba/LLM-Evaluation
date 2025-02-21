import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LoggerConfig:
    @staticmethod
    def setup_logger(name: str, log_dir: Path, process_id: int = None) -> logging.Logger:
        """
        Setup a logger with file and console handlers.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            process_id: Optional process ID for parallel processing
        """
        # Create logs directory
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        logger.handlers = []

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File handler - separate file for each process if process_id provided
        if process_id is not None:
            log_file = log_dir / f"process_{process_id}.log"
        else:
            log_file = log_dir / "main.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger
