# base_processor.py
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
from constants import ProcessingConstants


class BaseProcessor:
    def __init__(
            self,
            data_dir: str,
            num_workers: int = ProcessingConstants.DEFAULT_NUM_WORKERS,
            batch_size: int = ProcessingConstants.DEFAULT_BATCH_SIZE,
            temp_dir: str = None,
            token: str = None
    ):
        """Initialize the base processor."""
        # Initialize basic attributes first
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.token = token
        self.logger = None

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Set up logger after data_dir is created
        self.setup_logger()
        
        # Set up temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = self.data_dir / ProcessingConstants.TEMP_DIR_NAME

        self.logger.info(f"Creating temporary directory: {self.temp_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialization complete for {self.__class__.__name__}")

    def setup_logger(self):
        """Set up logging configuration."""
        if self.logger is not None:
            return

        try:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)

            # Remove any existing handlers
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

            # Add handlers
            log_file = self.data_dir / 'processing.log'

            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            
            self.logger.info(f"Logger initialized for {self.__class__.__name__}")
            self.logger.info(f"Log file created at: {log_file}")
            
        except Exception as e:
            # Fallback to basic console logging if file logging fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.error(f"Failed to set up file logging: {e}")

    def cleanup_temp_dir(self, force: bool = False) -> bool:
        """Clean up temporary directory."""
        try:
            if force:
                for file in self.temp_dir.glob(f"*{ProcessingConstants.PARQUET_EXTENSION}"):
                    file.unlink()
                self.logger.info("Cleaned up temporary directory")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary directory: {e}")
            return False

    def cleanup_file(self, file_path: Path, success: bool = True):
        """
        Clean up a temporary file.
        
        Args:
            file_path: Path to the file to clean up
            success: If True, delete the file. If False, keep it for debugging.
        """
        if not hasattr(self, 'logger'):
            return
            
        try:
            if file_path and file_path.exists():
                if success:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up successfully processed file: {file_path}")
                else:
                    self.logger.info(f"Keeping failed file for debugging: {file_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning up file {file_path}: {e}")

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            # Only force cleanup on destruction if environment variable is set
            force_cleanup = os.getenv('FORCE_CLEANUP', 'false').lower() == 'true'
            self.cleanup_temp_dir(force=force_cleanup)
        except:
            # Suppress errors during deletion
            pass