# uploader.py
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import os

from huggingface_hub import HfApi
from logger_config import LoggerConfig
from constants import ProcessingConstants
from config.get_config import Config


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
            # Upload full schema
            self.logger.info("Starting full schema upload...")
            full_schema_dir = self.data_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME
            self._upload_schema_dir(full_schema_dir, ProcessingConstants.OUTPUT_REPO)

            # Upload lean schema
            self.logger.info("\nStarting lean schema upload...")
            lean_schema_dir = self.data_dir / ProcessingConstants.LEAN_SCHEMA_DIR_NAME
            self._upload_schema_dir(lean_schema_dir, ProcessingConstants.OUTPUT_REPO_LEAN)

        except Exception as e:
            self.logger.error(f"Error in upload_all: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _ensure_repo_exists(self, repo_name: str):
        """Create repository if it doesn't exist."""
        try:
            self.api.create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            self.logger.info(f"Ensured repository {repo_name} exists")
        except Exception as e:
            self.logger.error(f"Error creating repository {repo_name}: {e}")
            raise

    def _upload_schema_dir(self, schema_dir: Path, repo_name: str):
        """Upload a schema directory to HuggingFace."""
        try:
            if not schema_dir.exists():
                self.logger.warning(f"Directory not found: {schema_dir}")
                return

            # Ensure repository exists
            self._ensure_repo_exists(repo_name)

            # Get model directories (skip logs directory)
            model_dirs = [
                d for d in schema_dir.iterdir() 
                if d.is_dir() and d.name != ProcessingConstants.LOGS_DIR_NAME
            ]
            
            if not model_dirs:
                self.logger.warning(f"No model directories found in {schema_dir}")
                return

            total_size_gb = sum(self._get_dir_size(d) for d in model_dirs)
            self.logger.info(f"Found {len(model_dirs)} model directories to upload (Total size: {total_size_gb:.2f}GB)")

            # Upload each model directory with progress bar
            for model_dir in tqdm(model_dirs, desc="Uploading model directories"):
                try:

                    # Count files for progress reporting
                    file_count = sum(1 for _ in model_dir.rglob('*') if _.is_file())
                    self.logger.info(f"Found {file_count} files in {model_dir.name}")

                    self.api.upload_folder(
                        folder_path=str(model_dir),
                        repo_id=repo_name,
                        repo_type="dataset",
                        path_in_repo=model_dir.name
                    )
                    self.logger.info(f"Successfully uploaded {model_dir.name}")

                except Exception as e:
                    import traceback
                    self.logger.error(f"Error uploading directory {model_dir.name}: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

            self.logger.info(f"Completed uploading all directories (Total: {total_size_gb:.2f}GB)")

        except Exception as e:
            self.logger.error(f"Error uploading schema directory {schema_dir}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")


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
