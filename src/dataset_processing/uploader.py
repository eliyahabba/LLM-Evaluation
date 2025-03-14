# uploader.py
import concurrent.futures
import os
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from huggingface_hub import HfApi
from tqdm import tqdm

from config.get_config import Config
from constants import ProcessingConstants
from logger_config import LoggerConfig


@dataclass
class UploadTask:
    """Represents a single upload task."""
    path: Path  # Local path to file/directory to upload
    repo_path: str  # Target path in the repo
    repo_name: str  # Repository name
    is_file: bool = False  # True for individual files, False for directories


class DatasetUploader:
    """Upload processed datasets to HuggingFace."""

    def __init__(self, data_dir: Path):
        """Initialize the dataset uploader."""
        self.data_dir = data_dir

        # Get HF token from config
        config = Config()
        self.token = config.config_values.get("hf_access_token")
        if not self.token:
            raise ValueError("HuggingFace access token not found in config")

        self.api = HfApi(token=self.token)
        self.logger = LoggerConfig.setup_logger(
            "DatasetUploader",
            self.data_dir / ProcessingConstants.LOGS_DIR_NAME
        )

    def _get_dir_size(self, dir_path: Path) -> float:
        """Get directory size in GB."""
        total_size = 0
        for dirpath, _, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024 * 1024)  # Convert to GB

    def upload_all(self):
        """Upload both full and lean schema datasets."""
        try:
            # Create a new logger for this process
            logger = LoggerConfig.setup_logger(
                "DatasetUploader",
                self.data_dir / ProcessingConstants.LOGS_DIR_NAME,
                process_id=os.getpid()
            )

            # Upload full schema
            logger.info("Starting full schema upload...")
            full_schema_dir = self.data_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME
            self._upload_schema_dir(full_schema_dir, ProcessingConstants.OUTPUT_REPO)

            # Upload lean schema
            logger.info("\nStarting lean schema upload...")
            lean_schema_dir = self.data_dir / ProcessingConstants.LEAN_SCHEMA_DIR_NAME
            self._upload_schema_dir(lean_schema_dir, ProcessingConstants.OUTPUT_REPO_LEAN)

        except Exception as e:
            logger.error(f"Error in upload_all: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _ensure_repo_exists(self, repo_name: str, logger=None):
        """Create repository if it doesn't exist."""
        try:
            if logger is None:
                logger = self.logger

            self.api.create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            logger.info(f"Ensured repository {repo_name} exists")
        except Exception as e:
            logger.error(f"Error creating repository {repo_name}: {e}")
            raise

    def _upload_schema_dir(self, schema_dir: Path, repo_name: str):
        """Upload a schema directory to HuggingFace."""
        try:
            # Create a new logger for this process
            logger = LoggerConfig.setup_logger(
                "DatasetUploader",
                self.data_dir / ProcessingConstants.LOGS_DIR_NAME,
                process_id=os.getpid()
            )

            if not schema_dir.exists():
                logger.warning(f"Directory not found: {schema_dir}")
                return

            # Ensure repository exists
            self._ensure_repo_exists(repo_name, logger)

            # Get model directories (skip logs directory)
            model_dirs = [
                d for d in schema_dir.iterdir()
                if d.is_dir() and d.name != ProcessingConstants.LOGS_DIR_NAME
            ]

            if not model_dirs:
                logger.warning(f"No model directories found in {schema_dir}")
                return

            # Collect all upload tasks
            upload_tasks: List[UploadTask] = []
            for model_dir in model_dirs:
                for lang_dir in model_dir.iterdir():
                    if not lang_dir.is_dir():
                        continue

                    # Get relative path components
                    model_name = model_dir.name
                    lang_name = lang_dir.name

                    # Special handling for English
                    if lang_dir.name == "en":
                        for shots_dir in lang_dir.iterdir():
                            if not shots_dir.is_dir():
                                continue
                            shots_name = shots_dir.name
                            for file_path in shots_dir.glob("*.parquet"):
                                upload_tasks.append(UploadTask(
                                    path=file_path,
                                    repo_path=f"{model_name}/{lang_name}/{shots_name}/{file_path.name}",
                                    repo_name=repo_name,
                                    is_file=True
                                ))
                    else:
                        for shots_dir in lang_dir.iterdir():
                            if not shots_dir.is_dir():
                                continue
                            upload_tasks.append(UploadTask(
                                path=shots_dir,
                                repo_path=f"{model_name}/{lang_name}/{shots_dir.name}",
                                repo_name=repo_name
                            ))
            total_tasks = len(upload_tasks)
            logger.info(f"Found {total_tasks} items to upload")

            # Process uploads in parallel
            successful_uploads = []
            failed_uploads = []
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=ProcessingConstants.DEFAULT_NUM_WORKERS) as executor:
                futures = {
                    executor.submit(self._process_upload_task, task): task
                    for task in upload_tasks
                }

                completed = 0
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Uploading files/directories"
                ):
                    task = futures[future]
                    try:
                        success = future.result()
                        completed += 1
                        if success:
                            successful_uploads.append(task.repo_path)
                            logger.info(
                                f"Successfully uploaded {task.path.name} "
                                f"({completed}/{total_tasks} completed)"
                            )
                        else:
                            failed_uploads.append(task.repo_path)
                            logger.error(f"Failed to upload {task.path.name}")
                    except Exception as e:
                        failed_uploads.append(task.repo_path)
                        logger.error(f"Error uploading {task.path.name}: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")

                # Log summary of uploads
                logger.info("\nUpload Summary:")
                logger.info(f"Total files attempted: {total_tasks}")
                logger.info(f"Successfully uploaded: {len(successful_uploads)}")
                logger.info(f"Failed uploads: {len(failed_uploads)}")
                
                if successful_uploads:
                    logger.info("\nSuccessfully uploaded files:")
                    for path in sorted(successful_uploads):
                        logger.info(f"  ✓ {path}")
                    
                if failed_uploads:
                    logger.error("\nFailed uploads:")
                    for path in sorted(failed_uploads):
                        logger.error(f"  ✗ {path}")

            return completed

        except Exception as e:
            logger.error(f"Error uploading schema directory {schema_dir}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _process_upload_task(self, task: UploadTask) -> bool:
        """Process a single upload task."""
        try:
            # Create a new logger for this process
            logger = LoggerConfig.setup_logger(
                "DatasetUploader",
                self.data_dir / ProcessingConstants.LOGS_DIR_NAME,
                process_id=os.getpid()
            )

            # Calculate size
            size_gb = (
                os.path.getsize(task.path) if task.is_file
                else self._get_dir_size(task.path)
            ) / (1024 * 1024 * 1024)

            max_retries = 3
            retry_delay = 60  # seconds
            rate_limit_delay = 3600  # 1 hour in seconds

            for attempt in range(max_retries):
                try:
                    if task.is_file:
                        logger.info(f"Uploading file: {task.path} ({size_gb:.2f}GB)")
                        self.api.upload_file(
                            path_or_fileobj=str(task.path),
                            path_in_repo=task.repo_path,
                            repo_id=task.repo_name,
                            repo_type="dataset"
                        )
                    else:
                        logger.info(f"Uploading directory: {task.path} ({size_gb:.2f}GB)")
                        self.api.upload_folder(
                            folder_path=str(task.path),
                            repo_id=task.repo_name,
                            repo_type="dataset",
                            path_in_repo=task.repo_path
                        )
                    
                    # Add delay between uploads to avoid rate limiting
                    time.sleep(1)  # 1 second delay between uploads
                    return True

                except Exception as e:
                    if "rate-limited" in str(e).lower():
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limited. Waiting {rate_limit_delay/3600:.1f} hours before retry {attempt + 1}/{max_retries}")
                            time.sleep(rate_limit_delay)
                        else:
                            logger.error(f"Failed after {max_retries} rate limit retries")
                            raise
                    else:
                        if attempt < max_retries - 1:
                            logger.warning(f"Upload failed. Retrying in {retry_delay} seconds ({attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                        else:
                            raise

        except Exception as e:
            logger.error(f"Error processing upload task for {task.path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-type', choices=['all', 'full', 'lean'], default='all',
                        help='Which schema to upload')
    args = parser.parse_args()

    uploader = DatasetUploader(Path(ProcessingConstants.OUTPUT_DATA_DIR))

    if args.schema_type == 'all':
        uploader.upload_all()
    elif args.schema_type == 'full':
        full_dir = Path(ProcessingConstants.OUTPUT_DATA_DIR) / ProcessingConstants.FULL_SCHEMA_DIR_NAME
        uploader._upload_schema_dir(full_dir, ProcessingConstants.OUTPUT_REPO)
    else:  # lean
        lean_dir = Path(ProcessingConstants.OUTPUT_DATA_DIR) / ProcessingConstants.LEAN_SCHEMA_DIR_NAME
        uploader._upload_schema_dir(lean_dir, ProcessingConstants.OUTPUT_REPO_LEAN)
