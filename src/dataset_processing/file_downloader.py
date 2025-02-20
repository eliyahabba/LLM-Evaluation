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
        self.processed_files_path = Path(self.data_dir) / ProcessingConstants.PROCESSED_FILES_RECORD
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
        """
        Get list of files that don't exist in the data directory.

        Args:
            file_extension: File extension to filter by

        Returns:
            List of files that need to be downloaded
        """
        try:
            all_files = self.hf_api.list_repo_files(self.source_repo, repo_type="dataset")
            parquet_files = [f for f in all_files if f.endswith(file_extension)]

            # Filter out files that already exist
            new_files = sorted([f for f in parquet_files if not self.file_exists(Path(f).name)])

            self.logger.info(f"Found {len(parquet_files)} total files")
            self.logger.info(f"Found {len(new_files)} new files to process")

            return new_files

        except Exception as e:
            self.logger.error(f"Error listing repository files: {e}")
            return []

    def download_file(self, file_path: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Download a file from HuggingFace to temporary directory.

        Args:
            file_path: Path of the file in the repository

        Returns:
            Tuple of (temporary file path, final file path) or (None, None) if download fails
        """
        try:
            filename = Path(file_path).name
            temp_path = self.temp_dir / filename
            final_path = self.data_dir / filename

            # First check if file exists in final location
            if final_path.exists():
                self.logger.info(f"File already exists in final location: {final_path}")
                return None, final_path

            # Then check if file exists in temp location
            if temp_path.exists():
                self.logger.info(f"File already exists in temp location: {temp_path}")
                return temp_path, final_path

            # Download to temporary location
            self.logger.info(f"Downloading {file_path} to temporary location...")
            hf_hub_download(
                repo_id=self.source_repo,
                filename=file_path,
                repo_type="dataset",
                local_dir=self.temp_dir,
                token=self.token
            )

            if temp_path.exists():
                self.logger.info(f"Successfully downloaded: {file_path} to {temp_path}")
                return temp_path, final_path
            else:
                self.logger.error(f"Download completed but file not found: {temp_path}")
                return None, None

        except Exception as e:
            self.logger.error(f"Error downloading {file_path}: {e}")
            return None, None

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