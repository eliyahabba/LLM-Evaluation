# uploader.py
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from base_processor import BaseProcessor
from constants import ProcessingConstants
from config.get_config import Config


class HFUploader(BaseProcessor):
    def __init__(
        self,
        output_repo: str,
        data_dir: str,
        token: str = None,
        batch_size: int = ProcessingConstants.DEFAULT_BATCH_SIZE
    ):
        """Initialize the HuggingFace uploader."""
        super().__init__(data_dir=data_dir, token=token, batch_size=batch_size)
        self.output_repo = output_repo
        self.hf_api = HfApi(token=self.token)

    def upload_file(self, file_path: Path) -> bool:
        """Upload a single file to HuggingFace."""
        try:
            self.logger.info(f"Uploading {file_path} to {self.output_repo}")
            self.hf_api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=self.output_repo,
                repo_type="dataset"
            )
            self.logger.info(f"Successfully uploaded {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading {file_path}: {e}")
            return False

    def upload_directory(self, directory: Path) -> bool:
        """Upload all files in a directory to HuggingFace."""
        try:
            success = True
            for file_path in directory.glob(f"*{ProcessingConstants.PARQUET_EXTENSION}"):
                if not self.upload_file(file_path):
                    success = False
            return success
        except Exception as e:
            self.logger.error(f"Error uploading directory {directory}: {e}")
            return False


def main():
    """Main execution function for uploading processed files."""
    config = Config()
    token = config.config_values.get("hf_access_token", "")

    uploader = HFUploader(
        output_repo=ProcessingConstants.OUTPUT_REPO,
        data_dir=ProcessingConstants.OUTPUT_DATA_DIR,
        token=token
    )

    # Upload full schema files
    full_schema_dir = Path(ProcessingConstants.OUTPUT_DATA_DIR) / ProcessingConstants.FULL_SCHEMA_DIR_NAME
    if not uploader.upload_directory(full_schema_dir):
        uploader.logger.error("Failed to upload full schema files")

    # Upload lean schema files
    lean_schema_dir = Path(ProcessingConstants.OUTPUT_DATA_DIR) / ProcessingConstants.LEAN_SCHEMA_DIR_NAME
    if not uploader.upload_directory(lean_schema_dir):
        uploader.logger.error("Failed to upload lean schema files")


if __name__ == "__main__":
    main()
