# main.py
import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional

from config.get_config import Config
from constants import ProcessingConstants
from deduplication_processor import DeduplicationProcessor
from file_downloader import HFFileDownloader
from full_schema_processor import FullSchemaProcessor
from lean_schema_processor import LeanSchemaProcessor
from logger_config import LoggerConfig
from processed_files_manager import ProcessedFilesManager


def process_single_file_standalone(file_path: str, input_dir: str, output_dir: str, token: str) -> List[str]:
    """Process a single file in a separate process."""
    # Setup logger first thing
    logger = None
    try:
        # Setup logger for this process
        log_dir = Path(output_dir) / ProcessingConstants.LOGS_DIR_NAME
        logger = LoggerConfig.setup_logger(
            "processor",
            log_dir,
            process_id=os.getpid()
        )
        logger.info(f"Starting processing of file: {file_path}")

        # Create processors for this specific process
        full_schema_dir = Path(output_dir) / ProcessingConstants.FULL_SCHEMA_DIR_NAME
        lean_schema_dir = Path(output_dir) / ProcessingConstants.LEAN_SCHEMA_DIR_NAME

        downloader = HFFileDownloader(
            source_repo=ProcessingConstants.SOURCE_REPO,
            data_dir=input_dir,
            token=token
        )

        full_processor = FullSchemaProcessor(
            data_dir=str(full_schema_dir),
            token=token
        )

        lean_processor = LeanSchemaProcessor(
            data_dir=str(lean_schema_dir),
            token=token
        )

        # Process file
        local_path = downloader.download_file(file_path)
        if local_path is None:
            return []

        # Process full schema
        full_schema_files = full_processor.process_file(local_path)
        if not full_schema_files:
            logger.error(f"Full schema processing failed for {file_path}")
            return []

        # Process lean schema
        lean_schema_files = []
        for full_file in full_schema_files:
            lean_files = lean_processor.process_file(Path(full_file))
            if not lean_files:
                logger.error(f"Lean schema processing failed for {full_file}")
                return []
            lean_schema_files.extend(lean_files)

        # Mark as processed only if all steps succeeded
        if full_schema_files and lean_schema_files:
            processed_files_manager = ProcessedFilesManager(input_dir)
            processed_files_manager.mark_as_processed(local_path)
            logger.info(f"Process {os.getpid()} completed processing file: {file_path}")
            return full_schema_files + lean_schema_files

        return []

    except Exception as e:
        # Use logger if available, otherwise print
        if logger:
            logger.error(f"Error processing {file_path}: {e}")
        else:
            print(f"Error processing {file_path} (before logger setup): {e}")
        return []


def process_full_schema_standalone(file_path: str, input_dir: str, output_dir: str, token: str) -> List[str]:
    """Process a single file to full schema in a separate process."""
    logger = None
    try:
        # Setup logger
        log_dir = Path(output_dir) / ProcessingConstants.LOGS_DIR_NAME
        logger = LoggerConfig.setup_logger(
            "full_schema_processor",
            log_dir,
            process_id=os.getpid()
        )
        logger.info(f"Starting full schema processing of file: {file_path}")

        # Create processors
        full_schema_dir = Path(output_dir) / ProcessingConstants.FULL_SCHEMA_DIR_NAME
        
        downloader = HFFileDownloader(
            source_repo=ProcessingConstants.SOURCE_REPO,
            data_dir=input_dir,
            token=token
        )

        full_processor = FullSchemaProcessor(
            data_dir=str(full_schema_dir),
            token=token
        )

        # Process file
        local_path = downloader.download_file(file_path)
        if local_path is None:
            return []

        # Process full schema
        full_schema_files = full_processor.process_file(local_path)
        if not full_schema_files:
            logger.error(f"Full schema processing failed for {file_path}")
            return []

        # Mark both the original file and full schema files as processed
        input_files_manager = ProcessedFilesManager(input_dir)
        input_files_manager.mark_as_processed(local_path)
        
        full_schema_manager = ProcessedFilesManager(
            data_dir=output_dir,
            record_file=ProcessingConstants.FULL_SCHEMA_FILES_RECORD
        )
        for full_file in full_schema_files:
            full_schema_manager.mark_as_processed(Path(full_file))
            
        logger.info(f"Process {os.getpid()} completed full schema processing: {file_path}")
        
        return full_schema_files

    except Exception as e:
        if logger:
            logger.error(f"Error processing full schema for {file_path}: {e}")
        else:
            print(f"Error processing full schema for {file_path} (before logger setup): {e}")
        return []


class UnifiedDatasetProcessor:
    def __init__(
            self,
            source_repo: str,
            output_repo: str,
            input_dir: str,
            output_dir: str,
            num_workers: int = ProcessingConstants.DEFAULT_NUM_WORKERS,
            batch_size: int = ProcessingConstants.DEFAULT_BATCH_SIZE,
            token: Optional[str] = None
    ):
        """Initialize the unified dataset processor."""
        # Create directories
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        full_schema_dir = output_dir_path / ProcessingConstants.FULL_SCHEMA_DIR_NAME
        lean_schema_dir = output_dir_path / ProcessingConstants.LEAN_SCHEMA_DIR_NAME

        for dir_path in [input_dir_path, output_dir_path, full_schema_dir, lean_schema_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save parameters
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.token = token
        self.num_workers = num_workers

        # Initialize processors
        self.downloader = HFFileDownloader(
            source_repo=source_repo,
            data_dir=input_dir,
            token=token,
            batch_size=batch_size
        )

        self.deduplicator = DeduplicationProcessor(data_dir=output_dir)
        self.logger = self.downloader.logger

    def process_all_files(self):
        """Process all new files in parallel."""
        try:
            new_files = self.downloader.get_new_files()
            if not new_files:
                self.logger.info("No new files to process")
                return

            all_processed_files = []

            # Process files in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_file_standalone,
                        file_path,
                        self.input_dir,
                        self.output_dir,
                        self.token
                    ): file_path
                    for file_path in new_files
                }

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    try:
                        processed_files = future.result()
                        all_processed_files.extend(processed_files)
                        self.logger.info(f"Processed {file_path} into {len(processed_files)} files")
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")

            # Deduplicate results
            if all_processed_files:
                self.logger.info(f"Starting deduplication of {len(all_processed_files)} files")
                if not self.deduplicator.deduplicate_files(set(all_processed_files)):
                    self.logger.error("Deduplication failed")

        except Exception as e:
            self.logger.error(f"Error in process_all_files: {e}")

    def process_full_schema_files(self):
        """Process all new files to full schema in parallel."""
        try:
            new_files = self.downloader.get_new_files()
            if not new_files:
                self.logger.info("No new files to process")
                return []

            all_processed_files = []

            # Process files in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        process_full_schema_standalone,
                        file_path,
                        self.input_dir,
                        self.output_dir,
                        self.token
                    ): file_path
                    for file_path in new_files
                }

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    try:
                        processed_files = future.result()
                        all_processed_files.extend(processed_files)
                        self.logger.info(f"Created full schema for {file_path} into {len(processed_files)} files")
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")

            return all_processed_files

        except Exception as e:
            self.logger.error(f"Error in process_full_schema_files: {e}")
            return []


if __name__ == "__main__":
    config = Config()
    token = config.config_values.get("hf_access_token", "")

    processor = UnifiedDatasetProcessor(
        source_repo=ProcessingConstants.SOURCE_REPO,
        output_repo=ProcessingConstants.OUTPUT_REPO,
        input_dir=ProcessingConstants.INPUT_DATA_DIR,
        output_dir=ProcessingConstants.OUTPUT_DATA_DIR,
        num_workers=ProcessingConstants.DEFAULT_NUM_WORKERS,
        batch_size=ProcessingConstants.DEFAULT_BATCH_SIZE,
        token=token
    )
    
    # Only process full schema files
    processor.process_full_schema_files()
