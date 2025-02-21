from pathlib import Path
from filelock import FileLock
import logging
import os

from constants import ProcessingConstants

class ProcessedFilesManager:
    """Manages the record of processed files."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.processed_files_path = self.data_dir / ProcessingConstants.PROCESSED_FILES_RECORD
        self.logger = logging.getLogger(__name__)
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self) -> set:
        """Load list of previously processed files."""
        if self.processed_files_path.exists():
            return set(self.processed_files_path.read_text().splitlines())
        return set()

    def mark_as_processed(self, file_path: Path) -> None:
        """Mark a file as processed by adding it to the processed files record."""
        try:
            lock_file = self.processed_files_path.with_suffix('.lock')
            
            with FileLock(lock_file):
                # Read existing processed files
                processed_files = set()
                if self.processed_files_path.exists():
                    with open(self.processed_files_path, 'r') as f:
                        processed_files = {line.strip() for line in f}
                
                # Only add if not already processed
                if str(file_path) not in processed_files:
                    # Verify file exists and is complete
                    if file_path.exists() and file_path.stat().st_size > 0:
                        with open(self.processed_files_path, 'a') as f:
                            f.write(f"{file_path}\n")
                            f.flush()  # Force write to disk
                            os.fsync(f.fileno())  # Ensure it's written to disk
                        
                        self.logger.info(f"Marked {file_path} as processed")
                        self.processed_files.add(str(file_path))
                    else:
                        self.logger.warning(f"File {file_path} not found or empty, not marking as processed")
                else:
                    self.logger.debug(f"File {file_path} already marked as processed")
            
        except Exception as e:
            self.logger.error(f"Error marking file as processed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def is_processed(self, file_path: str) -> bool:
        """Check if a file has been processed."""
        return str(file_path) in self.processed_files

    def get_unprocessed_files(self, all_files: list) -> list:
        """Get list of files that haven't been processed yet."""
        return [f for f in all_files if not self.is_processed(f)] 