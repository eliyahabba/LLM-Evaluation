# main.py
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Optional

from config.get_config import Config
from constants import ProcessingConstants
from deduplication_processor import DeduplicationProcessor
from file_downloader import HFFileDownloader
from full_schema_processor import FullSchemaProcessor
from lean_schema_processor import LeanSchemaProcessor
from uploader import HFUploader

config = Config()

TOKEN = config.config_values.get("hf_access_token", "")


class UnifiedDatasetProcessor:
    def __init__(
            self,
            source_repo: str,
            output_repo: str,
            data_dir: str,
            num_workers: int = 4,
            batch_size: int = 10000,
            temp_dir: Optional[str] = None,
            token: Optional[str] = None
    ):
        """Initialize the unified dataset processor."""
        self.logger = None

        # Create directories
        data_dir_path = Path(data_dir)
        full_schema_dir = data_dir_path / "full_schema"
        lean_schema_dir = data_dir_path / "lean_schema"
        temp_dir = data_dir_path / "temp"  # Single temp directory at root level

        # Create all directories
        full_schema_dir.mkdir(parents=True, exist_ok=True)
        lean_schema_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        common_args = {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "temp_dir": str(temp_dir),  # Pass the same temp_dir to all processors
            "token": token
        }

        try:
            self.downloader = HFFileDownloader(source_repo, data_dir=data_dir, **common_args)
            self.full_processor = FullSchemaProcessor(data_dir=str(full_schema_dir), **common_args)
            self.lean_processor = LeanSchemaProcessor(data_dir=str(lean_schema_dir), **common_args)
            self.uploader = HFUploader(output_repo, data_dir=data_dir, **common_args)
            self.deduplicator = DeduplicationProcessor(data_dir=data_dir, **common_args)

            self.logger = self.downloader.logger
            self.files_processed_this_run = set()

        except Exception as e:
            logging.error(f"Failed to initialize UnifiedDatasetProcessor: {e}")
            raise

    def process_single_file(self, file_path: str) -> List[str]:
        """Process a single file through the entire pipeline."""
        temp_path, final_path = self.downloader.download_file(file_path)

        if temp_path is None:
            return []

        try:
            self.logger.info(f"Processing file: {temp_path}")

            # Process full schema first
            full_schema_files = self.full_processor.process_file(temp_path)

            # Process lean schema from full schema files
            lean_schema_files = []
            for full_file in full_schema_files:
                lean_schema_files.extend(self.lean_processor.process_file(Path(full_file)))

            # Track processed files
            processed_files = full_schema_files + lean_schema_files
            self.files_processed_this_run.update(processed_files)

            # Mark original file as processed
            self.downloader._mark_as_processed(file_path)

            return processed_files

        except Exception as e:
            self.logger.error(f"Error processing {temp_path}: {e}")
            return []

    def process_all_files(self):
        """Process all new files from the source repository."""
        try:
            # Get list of new files
            new_files = self.downloader.get_new_files()[:1]
            if not new_files:
                self.downloader.logger.info("No new files to process")
                return

            # Process files in parallel
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.downloader.num_workers
            ) as executor:
                futures = {
                    executor.submit(self.process_single_file, f): f
                    for f in new_files
                }

                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    try:
                        processed_files = future.result()
                        self.downloader.logger.info(
                            f"Processed {file_path} into {len(processed_files)} output files"
                        )
                    except Exception as e:
                        self.downloader.logger.error(f"Error processing {file_path}: {e}")

            # After all files are processed, deduplicate only new files
            if self.files_processed_this_run:
                self.logger.info("Starting deduplication of newly processed files")
                if not self.deduplicator.deduplicate_files(self.files_processed_this_run):
                    self.logger.error("Deduplication failed")

        except Exception as e:
            self.logger.error(f"Error in process_all_files: {e}")
        # finally:
        #     # Cleanup temp directory
        #     self.downloader.cleanup_temp_dir(force=True)


if __name__ == "__main__":
    processor = UnifiedDatasetProcessor(
        source_repo=ProcessingConstants.SOURCE_REPO,
        output_repo=ProcessingConstants.OUTPUT_REPO,
        data_dir=ProcessingConstants.DEFAULT_DATA_DIR,
        num_workers=ProcessingConstants.DEFAULT_NUM_WORKERS,
        batch_size=ProcessingConstants.DEFAULT_BATCH_SIZE,
        token=TOKEN
    )
    processor.process_all_files()
