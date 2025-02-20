# main.py
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Optional
import os

from config.get_config import Config
from constants import ProcessingConstants
from file_downloader import HFFileDownloader

config = Config()
TOKEN = config.config_values.get("hf_access_token", "")

def download_single_file(file_path: str, input_dir: str, token: str) -> bool:
    """Download a single file from HuggingFace."""
    try:
        downloader = HFFileDownloader(
            ProcessingConstants.SOURCE_REPO,
            data_dir=input_dir,
            token=token
        )
        
        # Download file
        local_path = downloader.download_file(file_path)
        if local_path is None:
            return False
            
        # Mark as processed and log success
        downloader._mark_as_processed(local_path)
        logging.info(f"Process {os.getpid()} downloaded file: {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading {file_path}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

class DatasetDownloader:
    def __init__(
            self,
            source_repo: str,
            input_dir: str,
            num_workers: int = ProcessingConstants.DEFAULT_NUM_WORKERS,
            token: Optional[str] = None
    ):
        """Initialize the dataset downloader."""
        self.input_dir = input_dir
        self.token = token
        self.num_workers = num_workers
        
        # Create input directory
        input_dir_path = Path(input_dir)
        input_dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize downloader for getting file list
        self.downloader = HFFileDownloader(source_repo, data_dir=input_dir, token=token)
        self.logger = self.downloader.logger

    def download_all_files(self):
        """Download all new files in parallel using ProcessPoolExecutor."""
        try:
            new_files = self.downloader.get_new_files()
            if not new_files:
                self.logger.info("No new files to download")
                return

            successful_downloads = 0
            failed_downloads = 0
            
            # Download files in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit each file to a separate process
                futures = {
                    executor.submit(
                        download_single_file, 
                        file_path,
                        self.input_dir,
                        self.token
                    ): file_path 
                    for file_path in new_files
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    try:
                        success = future.result()
                        if success:
                            successful_downloads += 1
                            self.logger.info(f"Successfully downloaded: {file_path}")
                        else:
                            failed_downloads += 1
                            self.logger.warning(f"Failed to download: {file_path}")
                    except Exception as e:
                        failed_downloads += 1
                        self.logger.error(f"Error downloading {file_path}: {e}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")

            self.logger.info(f"Download complete. Successful: {successful_downloads}, Failed: {failed_downloads}")

        except Exception as e:
            self.logger.error(f"Error in download_all_files: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    downloader = DatasetDownloader(
        source_repo=ProcessingConstants.SOURCE_REPO,
        input_dir=ProcessingConstants.INPUT_DATA_DIR,
        num_workers=ProcessingConstants.DEFAULT_NUM_WORKERS,
        token=TOKEN
    )
    downloader.download_all_files()
