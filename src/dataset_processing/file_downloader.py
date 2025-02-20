# file_downloader.py
from pathlib import Path
from typing import Optional, List, Tuple

from huggingface_hub import HfApi, hf_hub_download

from base_processor import BaseProcessor
from constants import ProcessingConstants, DatasetRepos


class HFFileDownloader(BaseProcessor):
    def __init__(
            self,
            source_repo: str,
            **kwargs
    ):
        """
        Initialize the HuggingFace file downloader.

        Args:
            source_repo (str): HuggingFace repo ID for source files
            **kwargs: Additional arguments passed to BaseProcessor
        """
        super().__init__(**kwargs)
        self.source_repo = source_repo
        self.hf_api = HfApi(token=self.token)
        
        # Set up directories
        self.temp_dir = self.data_dir / ProcessingConstants.TEMP_DIR_NAME
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load record of processed files
        self.processed_files_path = self.data_dir / ProcessingConstants.PROCESSED_FILES_RECORD
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self) -> set:
        """Load list of previously processed files."""
        if self.processed_files_path.exists():
            return set(self.processed_files_path.read_text().splitlines())
        return set()

    def _mark_as_processed(self, file_path: str):
        """Mark a file as processed."""
        self.processed_files.add(file_path)
        with open(self.processed_files_path, 'a') as f:
            f.write(f"{file_path}\n")

    def file_exists(self, filename: str) -> bool:
        """
        Check if a file already exists in the data directory.

        Args:
            filename: Name of the file to check

        Returns:
            bool: True if file exists
        """
        return (self.data_dir / filename).exists()

    def get_new_files(self, file_extension: str = '.parquet') -> List[str]:
        """Get list of files from HF that haven't been processed yet."""
        try:
            # Get all files from HF
            all_files = self.hf_api.list_repo_files(self.source_repo, repo_type="dataset")
            parquet_files = [f for f in all_files if f.endswith(file_extension)]
            
            # Filter out files we've already processed
            new_files = [f for f in parquet_files if f not in self.processed_files]
            
            self.logger.info(f"Found {len(parquet_files)} total files")
            self.logger.info(f"Found {len(new_files)} new files to process")
            
            return sorted(new_files)
            
        except Exception as e:
            self.logger.error(f"Error listing repository files: {e}")
            return []

    def download_file(self, file_path: str) -> Optional[Path]:
        """Download or use existing file for processing."""
        try:
            filename = Path(file_path).name
            data_path = self.data_dir / filename
            temp_path = self.temp_dir / filename
            
            # If file exists in data_dir, use it
            if data_path.exists():
                self.logger.info(f"Using existing file: {data_path}")
                return data_path
                
            # Download to temp directory
            self.logger.info(f"Downloading {file_path} to temporary location...")
            try:
                hf_hub_download(
                    repo_id=self.source_repo,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=self.temp_dir,
                    token=self.token
                )
                
                # If download successful, move to data directory
                if temp_path.exists():
                    temp_path.replace(data_path)  # Atomic move
                    self.logger.info(f"Successfully moved file to: {data_path}")
                    return data_path
                    
            except Exception as download_error:
                self.logger.error(f"Download failed: {download_error}")
                if temp_path.exists():
                    temp_path.unlink()  # Clean up failed download
                return None
                
        except Exception as e:
            self.logger.error(f"Error handling file {file_path}: {e}")
            return None

    def move_to_final_location(self, temp_path: Path, final_path: Path) -> bool:
        """
        Move a file from temporary location to final location.

        Args:
            temp_path: Path to temporary file
            final_path: Path to final location

        Returns:
            bool: True if move was successful
        """
        try:
            temp_path.rename(final_path)
            self.logger.info(f"Moved file to final location: {final_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error moving file to final location: {e}")
            return False