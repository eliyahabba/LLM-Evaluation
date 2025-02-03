import os
import time
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm


class HFHierarchicalUploader:
    def __init__(self, local_dir: str, repo_name: str, token: str, max_workers: int = 4):
        self.local_dir = Path(local_dir)
        self.repo_name = repo_name
        self.token = token
        self.max_workers = max_workers
        self.api = HfApi(token=token)
        self.setup_logger()

    def setup_logger(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = logging.getLogger('HFUploader')
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(f'hf_upload_{timestamp}.log')
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_existing_files(self) -> List[str]:
        """Get list of existing files in the repository"""
        try:
            files = self.api.list_repo_files(repo_id=self.repo_name, repo_type="dataset")
            return files
        except Exception as e:
            self.logger.error(f"Error listing repository files: {e}")
            return []

    def collect_local_files(self) -> List[Path]:
        """Collect all parquet files with their relative paths"""
        files = []
        for file_path in self.local_dir.rglob("*.parquet"):
            files.append(file_path)
        return files

    def upload_file(self, file_path: Path) -> Optional[str]:
        """Upload a single file while maintaining directory structure"""
        try:
            # Calculate relative path to maintain directory structure
            relative_path = file_path.relative_to(self.local_dir)

            # Add small delay between uploads to avoid rate limiting
            time.sleep(1)

            self.api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(relative_path),
                repo_id=self.repo_name,
                repo_type="dataset",
                token=self.token
            )

            return str(relative_path)
        except Exception as e:
            self.logger.error(f"Error uploading {file_path}: {e}")
            return None

    def upload_files(self):
        """Upload files in parallel while maintaining structure"""
        self.logger.info(f"Starting upload to {self.repo_name}")

        # Ensure repository exists
        try:
            create_repo(
                repo_id=self.repo_name,
                token=self.token,
                private=True,
                repo_type="dataset"
            )
        except Exception as e:
            self.logger.info(f"Repository already exists or error occurred: {e}")

        # Get existing files
        existing_files = self.get_existing_files()
        self.logger.info(f"Found {len(existing_files)} existing files in repository")

        # Collect local files
        local_files = self.collect_local_files()
        self.logger.info(f"Found {len(local_files)} local files to process")

        # Filter out already uploaded files
        files_to_upload = []
        for file_path in local_files:
            relative_path = str(file_path.relative_to(self.local_dir))
            if relative_path not in existing_files:
                files_to_upload.append(file_path)

        self.logger.info(f"Uploading {len(files_to_upload)} new files")

        if not files_to_upload:
            self.logger.info("No new files to upload")
            return

        # Upload files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self.upload_file, files_to_upload),
                total=len(files_to_upload),
                desc="Uploading files"
            ))

        # Log results
        successful = [r for r in results if r is not None]
        failed = len(results) - len(successful)

        self.logger.info(f"Upload completed:")
        self.logger.info(f"- Successfully uploaded: {len(successful)} files")
        if failed > 0:
            self.logger.info(f"- Failed uploads: {failed} files")


if __name__ == "__main__":
    # Configuration
    LOCAL_DIR = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split"
    REPO_NAME = "eliyahabba/llm-evaluation-analysis-split2"

    from config.get_config import Config

    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
    MAX_WORKERS = 4  # Adjust based on your needs and HF rate limits

    # Initialize and run uploader
    uploader = HFHierarchicalUploader(
        local_dir=LOCAL_DIR,
        repo_name=REPO_NAME,
        token=TOKEN,
        max_workers=MAX_WORKERS
    )

    uploader.upload_files()