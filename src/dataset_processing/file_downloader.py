# file_downloader.py
from pathlib import Path
from typing import Optional, List

from huggingface_hub import HfApi, hf_hub_download

from base_processor import BaseProcessor
from constants import ProcessingConstants
from processed_files_manager import ProcessedFilesManager


class HFFileDownloader(BaseProcessor):
    def __init__(
            self,
            source_repo: str,
            data_dir: str,
            token: str = None,
            batch_size: int = ProcessingConstants.DEFAULT_BATCH_SIZE
    ):
        """Initialize the HuggingFace file downloader."""
        super().__init__(data_dir=data_dir, token=token, batch_size=batch_size)
        self.source_repo = source_repo
        self.hf_api = HfApi(token=self.token)

        # Set up temp directory
        self.temp_dir = self.data_dir / ProcessingConstants.TEMP_DIR_NAME
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Use ProcessedFilesManager
        self.processed_files_manager = ProcessedFilesManager(self.data_dir)

    def get_new_files(self, file_extension: str = '.parquet') -> List[str]:
        """Get list of files from HF that haven't been processed yet."""
        try:
            # Get all files from HF
            all_files = self.hf_api.list_repo_files(self.source_repo, repo_type="dataset")
            parquet_files = [f for f in all_files if f.endswith(file_extension)]

            # Use ProcessedFilesManager to filter unprocessed files
            new_files = self.processed_files_manager.get_unprocessed_files(parquet_files)

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

            # Check if file is already processed
            if self.processed_files_manager.is_processed(str(data_path)):
                self.logger.info(f"File already processed: {data_path}")
                return None

            # If file exists but not processed, use it
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

    def cleanup_temp_dir(self, force: bool = False) -> bool:
        """
        Clean up temporary directory.
        
        Args:
            force: If True, delete all files regardless of status
        """
        try:
            if force:
                for file in self.temp_dir.glob(f"*{ProcessingConstants.PARQUET_EXTENSION}"):
                    file.unlink()
                self.logger.info("Cleaned up temporary directory")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary directory: {e}")
            return False
