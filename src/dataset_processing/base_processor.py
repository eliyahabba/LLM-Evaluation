# base_processor.py
from pathlib import Path

from constants import ProcessingConstants
from logger_config import LoggerConfig


class BaseProcessor:
    def __init__(
            self,
            data_dir: str,
            token: str = None,
            batch_size: int = ProcessingConstants.DEFAULT_BATCH_SIZE
    ):
        """
        Initialize the base processor.
        
        Args:
            data_dir: Directory for data files
            token: HuggingFace token for API access
            batch_size: Size of batches for processing
        """
        self.data_dir = Path(data_dir)
        self.token = token
        self.batch_size = batch_size

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        log_dir = self.data_dir / ProcessingConstants.LOGS_DIR_NAME
        self.logger = LoggerConfig.setup_logger(
            self.__class__.__name__,
            log_dir
        )

    def cleanup_file(self, file_path: Path, success: bool = True):
        """
        Clean up a file.
        
        Args:
            file_path: Path to the file to clean up
            success: If True, delete the file. If False, keep it for debugging.
        """
        try:
            if file_path and file_path.exists():
                if success:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up successfully processed file: {file_path}")
                else:
                    self.logger.info(f"Keeping failed file for debugging: {file_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning up file {file_path}: {e}")
