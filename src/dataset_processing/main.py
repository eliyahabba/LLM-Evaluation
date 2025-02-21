# main.py
import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional
import logging

from config.get_config import Config
from constants import ProcessingConstants
from deduplication_processor import DeduplicationProcessor
from file_downloader import HFFileDownloader
from full_schema_processor import FullSchemaProcessor
from lean_schema_processor import LeanSchemaProcessor
from logger_config import LoggerConfig
from processed_files_manager import ProcessedFilesManager


def setup_process_logger(name: str, output_dir: str) -> logging.Logger:
    """Setup logger for a standalone process."""
    log_dir = Path(output_dir) / ProcessingConstants.LOGS_DIR_NAME
    return LoggerConfig.setup_logger(
        name,
        log_dir,
        process_id=os.getpid()
    )


def process_files_in_parallel(
        processor_func,
        files_to_process: List[str],
        num_workers: int,
        logger: logging.Logger,
        **kwargs
) -> List[str]:
    """Process files in parallel using the given processor function."""
    all_processed_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(processor_func, file_path, **kwargs): file_path
            for file_path in files_to_process
        }

        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                processed_files = future.result()
                all_processed_files.extend(processed_files)
                logger.info(f"Processed {file_path} into {len(processed_files)} files")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    return all_processed_files


def process_full_schema_standalone(file_path: str, input_dir: str, output_dir: str, token: str) -> List[str]:
    """Process a single file to full schema in a separate process."""
    logger = None
    try:
        logger = setup_process_logger("full_schema_processor", output_dir)
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


def process_lean_schema_standalone(full_schema_file: str, output_dir: str, token: str) -> List[str]:
    """Process a single full schema file to lean schema in a separate process."""
    logger = None
    try:
        logger = setup_process_logger("lean_schema_processor", output_dir)
        logger.info(f"Starting lean schema processing of file: {full_schema_file}")

        # Create processor
        lean_schema_dir = Path(output_dir) / ProcessingConstants.LEAN_SCHEMA_DIR_NAME
        lean_processor = LeanSchemaProcessor(
            data_dir=str(lean_schema_dir),
            token=token
        )

        # Process file
        lean_schema_files = lean_processor.process_file(Path(full_schema_file))
        if not lean_schema_files:
            logger.error(f"Lean schema processing failed for {full_schema_file}")
            return []

        # Remove from full schema tracking after successful processing
        full_schema_manager = ProcessedFilesManager(
            data_dir=output_dir,
            record_file=ProcessingConstants.FULL_SCHEMA_FILES_RECORD
        )
        full_schema_manager.remove_from_processed(Path(full_schema_file))

        logger.info(f"Process {os.getpid()} completed lean schema processing: {full_schema_file}")
        return lean_schema_files

    except Exception as e:
        if logger:
            logger.error(f"Error processing lean schema for {full_schema_file}: {e}")
        else:
            print(f"Error processing lean schema for {full_schema_file} (before logger setup): {e}")
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

    def process_full_schema_files(self):
        """Process all new files to full schema in parallel."""
        try:
            new_files = self.downloader.get_new_files()
            if not new_files:
                self.logger.info("No new files to process")
                return []

            return process_files_in_parallel(
                process_full_schema_standalone,
                new_files,
                self.num_workers,
                self.logger,
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                token=self.token
            )

        except Exception as e:
            self.logger.error(f"Error in process_full_schema_files: {e}")
            return []

    def process_lean_schema_files(self):
        """Process all unprocessed full schema files to lean schema in parallel."""
        try:
            # Get files to process
            full_schema_manager = ProcessedFilesManager(
                data_dir=self.output_dir,
                record_file=ProcessingConstants.FULL_SCHEMA_FILES_RECORD
            )

            full_schema_dir = Path(self.output_dir) / ProcessingConstants.FULL_SCHEMA_DIR_NAME
            all_full_schema_files = list(full_schema_dir.glob(f"*{ProcessingConstants.PARQUET_EXTENSION}"))

            files_to_process = [
                str(f) for f in all_full_schema_files
                if full_schema_manager.is_processed(str(f))
            ]

            if not files_to_process:
                self.logger.info("No full schema files to process")
                return []

            self.logger.info(f"Found {len(files_to_process)} full schema files to process")

            # Process files
            processed_files = process_files_in_parallel(
                process_lean_schema_standalone,
                files_to_process,
                self.num_workers,
                self.logger,
                output_dir=self.output_dir,
                token=self.token
            )

            # Deduplicate results
            if processed_files:
                self.logger.info(f"Starting deduplication of {len(processed_files)} lean schema files")
                if not self.deduplicator.deduplicate_files(set(processed_files)):
                    self.logger.error("Deduplication failed")

            return processed_files

        except Exception as e:
            self.logger.error(f"Error in process_lean_schema_files: {e}")
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

    # Choose which process to run
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "lean":
        processor.process_lean_schema_files()
    else:
        processor.process_full_schema_files()
