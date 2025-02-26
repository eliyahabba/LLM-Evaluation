import concurrent.futures
import os
from pathlib import Path
import time
import shutil
from typing import Optional
import argparse
from tqdm import tqdm
import traceback
import pyarrow.parquet as pq

from constants import ProcessingConstants, ParquetConstants
from deduplication_processor import DeduplicationProcessor
from lean_schema_processor import LeanSchemaProcessor
from logger_config import LoggerConfig
from optimized_deduplication import OptimizedDeduplicationProcessor


class DatasetMerger:
    def __init__(
            self,
            data_dir: str,
            num_workers: int = ProcessingConstants.DEFAULT_NUM_WORKERS
    ):
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.logger = LoggerConfig.setup_logger("DatasetMerger", self.data_dir / ProcessingConstants.LOGS_DIR_NAME)

    def process_all(self, process_type: str = "all"):
        """
        Run the processing pipeline.
        
        Args:
            process_type: Type of processing to run ("all", "merge", or "lean")
        """
        try:
            if process_type in ["all", "merge"]:
                # 1. Merge full schema files
                self.logger.info("Starting full schema files merge and deduplication")
                start_time = time.time()
                self.merge_full_schema_files()
                merge_time = time.time() - start_time
                self.logger.info(f"Full schema merge completed and deduplication in {merge_time:.2f} seconds")

            if process_type in ["all", "lean"]:
                # 2. Create lean schema from deduplicated files
                self.logger.info("\nStarting lean schema creation")
                start_time = time.time()
                self.create_lean_schema()
                lean_time = time.time() - start_time
                self.logger.info(f"Lean schema creation completed in {lean_time:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Error in process_all: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def merge_full_schema_files(self):
        """Merge all full schema files in parallel and deduplicate each merged file."""
        try:
            full_schema_dir = self.data_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME

            # Collect all dataset directories that need merging
            merge_tasks = []
            for model_dir in full_schema_dir.glob("*"):
                if not model_dir.is_dir():
                    continue
                for lang_dir in model_dir.glob("*"):
                    if not lang_dir.is_dir():
                        continue
                    for shots_dir in lang_dir.glob("*"):
                        if not shots_dir.is_dir():
                            continue
                        for dataset_dir in shots_dir.glob("*"):
                            if not dataset_dir.is_dir() or dataset_dir.name.endswith('.parquet'):
                                continue
                            merge_tasks.append((model_dir, lang_dir, shots_dir, dataset_dir))

            if not merge_tasks:
                self.logger.info("No directories to merge")
                return

            self.logger.info(f"Found {len(merge_tasks)} directories to merge")

            # Initialize deduplicator once
            deduplicator = OptimizedDeduplicationProcessor(self.data_dir)

            # Process merges in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        self._merge_and_deduplicate_dataset,
                        model_dir,
                        lang_dir,
                        shots_dir,
                        dataset_dir,
                        deduplicator
                    ): (model_dir.name, lang_dir.name, shots_dir.name, dataset_dir.name)
                    for model_dir, lang_dir, shots_dir, dataset_dir in merge_tasks
                }

                for future in concurrent.futures.as_completed(futures):
                    model, lang, shots, dataset = futures[future]
                    try:
                        success = future.result()
                        if success:
                            self.logger.info(f"Successfully processed {model}/{lang}/{shots}/{dataset}")
                        else:
                            self.logger.error(f"Failed to process {model}/{lang}/{shots}/{dataset}")
                    except Exception as e:
                        self.logger.error(f"Error processing {model}/{lang}/{shots}/{dataset}: {e}")

        except Exception as e:
            self.logger.error(f"Error in merge_full_schema_files: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _merge_and_deduplicate_dataset(self, model_dir: Path, lang_dir: Path, shots_dir: Path, dataset_dir: Path, deduplicator: OptimizedDeduplicationProcessor) -> bool:
        """Merge files in a dataset directory and then deduplicate the merged file."""
        try:
            # First merge the files
            merged_path = self._merge_dataset_files(model_dir, lang_dir, shots_dir, dataset_dir)
            if not merged_path:
                return False

            # Then deduplicate the merged file
            success = deduplicator.deduplicate_files({merged_path})
            return success

        except Exception as e:
            self.logger.error(f"Error in merge_and_deduplicate_dataset: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _merge_dataset_files(self, model_dir: Path, lang_dir: Path, shots_dir: Path, dataset_dir: Path) -> Optional[Path]:
        """Merge all files in a dataset directory into a single parquet file."""
        try:
            parquet_files = list(dataset_dir.glob(f"*{ProcessingConstants.PARQUET_EXTENSION}"))
            if not parquet_files:
                return None

            # Add '_merged' suffix to the merged file
            merged_path = model_dir / lang_dir.name / shots_dir.name / f"{dataset_dir.name}_merged.parquet"
            merged_path.parent.mkdir(parents=True, exist_ok=True)

            writer = None
            open_files = []
            for input_file in parquet_files:
                pf = pq.ParquetFile(str(input_file))
                open_files.append(pf)
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            str(merged_path),
                            table.schema,
                            version=ParquetConstants.VERSION,
                            write_statistics=ParquetConstants.WRITE_STATISTICS
                        )
                    writer.write_table(table)
            if writer:
                writer.close()
            for pf in open_files:
                pf.close()
            shutil.rmtree(dataset_dir, ignore_errors=True)
            if os.path.exists(dataset_dir):
                self.logger.error(f"Failed to remove directory: {dataset_dir}")
                self.logger.info(f"Failed2 to remove directory: {dataset_dir}")
                os.rmdir(dataset_dir)
            # If the only remaining file is the .nfs file, try to remove it manually.
            self.logger.info(f"Cleaned up directory: {dataset_dir}")

            return merged_path

        except Exception as e:
            self.logger.error(f"Error merging files in {dataset_dir}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def deduplicate_lean_schema(self):
        """Deduplicate lean schema files using standard processor."""
        deduplicator = DeduplicationProcessor(self.data_dir, schema_type="lean")
        lean_schema_dir = self.data_dir / ProcessingConstants.LEAN_SCHEMA_DIR_NAME
        parquet_files = list(lean_schema_dir.glob("**/*.parquet"))
        deduplicator.deduplicate_files(set(str(f) for f in parquet_files))

    def create_lean_schema(self):
        """Create lean schema from deduplicated full schema files."""
        try:
            full_schema_dir = self.data_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME
            
            # Collect all deduplicated files
            parquet_files = []
            for model_dir in full_schema_dir.glob("*"):
                if not model_dir.is_dir() or model_dir.name == ProcessingConstants.LOGS_DIR_NAME:
                    continue
                for lang_dir in model_dir.glob("*"):
                    if not lang_dir.is_dir():
                        continue
                    for shots_dir in lang_dir.glob("*"):
                        if not shots_dir.is_dir():
                            continue
                        for file in shots_dir.glob("*.parquet"):
                            parquet_files.append(file)

            if not parquet_files:
                self.logger.info("No files to process for lean schema")
                return

            self.logger.info(f"Found {len(parquet_files)} files to process for lean schema")
            
            processor = LeanSchemaProcessor(data_dir=str(self.data_dir / ProcessingConstants.LEAN_SCHEMA_DIR_NAME))
            
            # Process files in parallel with better error handling
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._process_lean_file_safely, processor, f): f
                    for f in parquet_files
                }

                completed = 0
                failed = 0
                total = len(futures)

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total,
                    desc="Processing files"
                ):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            completed += 1
                            self.logger.info(
                                f"Successfully processed {file_path.name} "
                                f"({completed}/{total} completed, {failed} failed)"
                            )
                        else:
                            failed += 1
                            self.logger.error(
                                f"Failed to process {file_path.name} "
                                f"({completed}/{total} completed, {failed} failed)"
                            )
                    except Exception as e:
                        failed += 1
                        self.logger.error(f"Error processing {file_path.name}: {e}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")

                self.logger.info(
                    f"Lean schema creation completed: "
                    f"{completed} successful, {failed} failed, {total} total"
                )

        except Exception as e:
            self.logger.error(f"Error in create_lean_schema: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _process_lean_file_safely(self, processor: LeanSchemaProcessor, file_path: Path) -> bool:
        """Safely process a single file for lean schema creation."""
        try:
            result = processor.process_file(file_path)
            return bool(result)  # Return True if we got any processed files
        except Exception as e:
            # Log error but don't raise - let the process continue
            self.logger.error(f"Error processing file {file_path}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-type', choices=['all', 'merge', 'lean'], default='all',
                       help='Type of processing to run')
    args = parser.parse_args()

    processor = DatasetMerger(
        data_dir=ProcessingConstants.OUTPUT_DATA_DIR,
        num_workers=ProcessingConstants.DEFAULT_NUM_WORKERS
    )
    processor.process_all(args.process_type)
