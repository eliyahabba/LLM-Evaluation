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

            # Process each model directory
            for model_dir in tqdm(model_dirs, desc="Processing model directories"):
                try:
                    # For each language directory in the model
                    for lang_dir in model_dir.iterdir():
                        if not lang_dir.is_dir():
                            continue

                        # Special handling for English
                        if lang_dir.name == "en":
                            self._upload_english_dir(lang_dir, model_dir.name, repo_name)
                        else:
                            # For other languages, upload each shots directory
                            self._upload_language_dir(lang_dir, model_dir.name, repo_name)

                except Exception as e:
                    self.logger.error(f"Error processing model directory {model_dir.name}: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

            self.logger.info(f"Completed uploading all directories (Total: {total_size_gb:.2f}GB)")

        except Exception as e:
            self.logger.error(f"Error uploading schema directory {schema_dir}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _upload_english_dir(self, lang_dir: Path, model_name: str, repo_name: str):
        """Upload English files individually within each shots directory."""
        try:
            # For each shots directory
            for shots_dir in lang_dir.iterdir():
                if not shots_dir.is_dir():
                    continue

                # Upload each file in the shots directory separately
                for file_path in shots_dir.glob("*.parquet"):
                    try:
                        file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
                        self.logger.info(f"Uploading file: {file_path.name} ({file_size_gb:.2f}GB)")

                        # Maintain the same directory structure in the repo
                        repo_path = f"{model_name}/{lang_dir.name}/{shots_dir.name}"
                        
                        self.api.upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=f"{repo_path}/{file_path.name}",
                            repo_id=repo_name,
                            repo_type="dataset"
                        )
                        self.logger.info(f"Successfully uploaded {file_path.name}")

                    except Exception as e:
                        self.logger.error(f"Error uploading file {file_path}: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Error processing English directory: {e}")

    def _upload_language_dir(self, lang_dir: Path, model_name: str, repo_name: str):
        """Upload each shots directory for non-English languages."""
        try:
            # For each shots directory
            for shots_dir in lang_dir.iterdir():
                if not shots_dir.is_dir():
                    continue

                try:
                    dir_size_gb = self._get_dir_size(shots_dir)
                    self.logger.info(f"Uploading directory: {shots_dir.name} ({dir_size_gb:.2f}GB)")

                    # Maintain the same directory structure in the repo
                    repo_path = f"{model_name}/{lang_dir.name}/{shots_dir.name}"
                    
                    self.api.upload_folder(
                        folder_path=str(shots_dir),
                        repo_id=repo_name,
                        repo_type="dataset",
                        path_in_repo=repo_path
                    )
                    self.logger.info(f"Successfully uploaded {shots_dir.name}")

                except Exception as e:
                    self.logger.error(f"Error uploading shots directory {shots_dir}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error processing language directory {lang_dir.name}: {e}")


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
