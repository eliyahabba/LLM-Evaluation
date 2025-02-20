# main.py
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Optional
from multiprocessing import Process
import numpy as np

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
            input_dir: str,  # Directory for HF downloads
            output_dir: str,  # Directory for processed files
            num_workers: int = ProcessingConstants.DEFAULT_NUM_WORKERS,
            batch_size: int = ProcessingConstants.DEFAULT_BATCH_SIZE,
            token: Optional[str] = None
    ):
        """Initialize the unified dataset processor."""
        self.logger = None

        # Create directories
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        full_schema_dir = output_dir_path / ProcessingConstants.FULL_SCHEMA_DIR_NAME
        lean_schema_dir = output_dir_path / ProcessingConstants.LEAN_SCHEMA_DIR_NAME

        # Create all directories
        input_dir_path.mkdir(parents=True, exist_ok=True)
        full_schema_dir.mkdir(parents=True, exist_ok=True)
        lean_schema_dir.mkdir(parents=True, exist_ok=True)

        # Common arguments for all processors
        common_args = {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "token": token
        }

        try:
            # Downloader uses input directory
            self.downloader = HFFileDownloader(source_repo, data_dir=input_dir, **common_args)
            
            # Processors use output directory
            self.full_processor = FullSchemaProcessor(data_dir=str(full_schema_dir), **common_args)
            self.lean_processor = LeanSchemaProcessor(data_dir=str(lean_schema_dir), **common_args)
            self.uploader = HFUploader(output_repo, data_dir=output_dir, **common_args)
            self.deduplicator = DeduplicationProcessor(data_dir=output_dir, **common_args)

            self.logger = self.downloader.logger
            self.files_processed_this_run = set()

        except Exception as e:
            logging.error(f"Failed to initialize UnifiedDatasetProcessor: {e}")
            raise

    def process_single_file(self, file_path: str) -> List[str]:
        """Process a single file through the entire pipeline."""
        file_path = self.downloader.download_file(file_path)

        if file_path is None:
            return []

        try:
            self.logger.info(f"Processing file: {file_path}")

            # Process full schema first
            full_schema_files = self.full_processor.process_file(file_path)

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
            self.logger.error(f"Error processing {file_path}: {e}")
            return []

    def process_all_files(self):
        """Process all new files in parallel using ProcessPoolExecutor."""
        try:
            new_files = self.downloader.get_new_files()
            if not new_files:
                self.downloader.logger.info("No new files to process")
                return

            all_processed_files = []
            
            # Process each file in a separate process
            with concurrent.futures.ProcessPoolExecutor(max_workers=ProcessingConstants.DEFAULT_NUM_WORKERS) as executor:
                # Submit each file to a separate process
                futures = {
                    executor.submit(self.process_single_file, file_path): file_path 
                    for file_path in new_files
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    try:
                        processed_files = future.result()
                        all_processed_files.extend(processed_files)
                        self.logger.info(f"Processed {file_path} into {len(processed_files)} files")
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")

            # After all files are processed, deduplicate
            if all_processed_files:
                self.logger.info(f"Starting deduplication of {len(all_processed_files)} files")
                if not self.deduplicator.deduplicate_files(set(all_processed_files)):
                    self.logger.error("Deduplication failed")

        except Exception as e:
            self.logger.error(f"Error in process_all_files: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")


def process_batch_files(file_batch):
    processor = UnifiedDatasetProcessor(
        source_repo=ProcessingConstants.SOURCE_REPO,
        output_repo=ProcessingConstants.OUTPUT_REPO,
        input_dir=ProcessingConstants.INPUT_DATA_DIR,
        output_dir=ProcessingConstants.OUTPUT_DATA_DIR,
        num_workers=ProcessingConstants.DEFAULT_NUM_WORKERS,
        batch_size=ProcessingConstants.DEFAULT_BATCH_SIZE,
        token=TOKEN
    )
    
    # Process this batch of files
    for file in file_batch:
        processor.process_single_file(file)


if __name__ == "__main__":
    processor = UnifiedDatasetProcessor(
        source_repo=ProcessingConstants.SOURCE_REPO,
        output_repo=ProcessingConstants.OUTPUT_REPO,
        input_dir=ProcessingConstants.INPUT_DATA_DIR,
        output_dir=ProcessingConstants.OUTPUT_DATA_DIR,
        num_workers=ProcessingConstants.DEFAULT_NUM_WORKERS,
        batch_size=ProcessingConstants.DEFAULT_BATCH_SIZE,
        token=TOKEN
    )
    processor.process_all_files()
