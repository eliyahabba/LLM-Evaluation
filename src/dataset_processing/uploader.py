# uploader.py
import traceback
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import os
import concurrent.futures
from dataclasses import dataclass

from huggingface_hub import HfApi
from logger_config import LoggerConfig
from constants import ProcessingConstants
from config.get_config import Config


@dataclass
class UploadTask:
    """Represents a single upload task."""
    path: Path
    model_name: str
    repo_name: str
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

                    # Special handling for English
                    if lang_dir.name == "en":
                        for shots_dir in lang_dir.iterdir():
                            if not shots_dir.is_dir():
                                continue
                            for file_path in shots_dir.glob("*.parquet"):
                                upload_tasks.append(UploadTask(
                                    path=file_path,
                                    model_name=model_dir.name,
                                    repo_name=repo_name,
                                    is_file=True
                                ))
                    else:
                        for shots_dir in lang_dir.iterdir():
                            if not shots_dir.is_dir():
                                continue
                            upload_tasks.append(UploadTask(
                                path=shots_dir,
                                model_name=model_dir.name,
                                repo_name=repo_name
                            ))

            total_tasks = len(upload_tasks)
            logger.info(f"Found {total_tasks} items to upload")

            # Process uploads in parallel
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
                            logger.info(
                                f"Successfully uploaded {task.path.name} "
                                f"({completed}/{total_tasks} completed)"
                            )
                        else:
                            logger.error(f"Failed to upload {task.path.name}")
                    except Exception as e:
                        logger.error(f"Error uploading {task.path.name}: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")

            logger.info(f"Completed uploading {completed}/{total_tasks} items")

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

            # Get repo path components
            lang = task.path.parent.parent.name
            shots = task.path.parent.name
            
            if task.is_file:
                # For files: model/lang/shots/file.parquet
                repo_path = f"{task.model_name}/{lang}/{shots}"
                logger.info(f"Uploading file: {task.path.name} ({size_gb:.2f}GB)")
                self.api.upload_file(
                    path_or_fileobj=str(task.path),
                    path_in_repo=f"{repo_path}/{task.path.name}",
                    repo_id=task.repo_name,
                    repo_type="dataset"
                )
            else:
                # For directories: model/lang/shots
                repo_path = f"{task.model_name}/{lang}"
                logger.info(f"Uploading directory: {task.path.name} ({size_gb:.2f}GB)")
                self.api.upload_folder(
                    folder_path=str(task.path),
                    repo_id=task.repo_name,
                    repo_type="dataset",
                    path_in_repo=repo_path
                )
            return True

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
