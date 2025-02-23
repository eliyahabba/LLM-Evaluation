import shutil
from pathlib import Path
import pyarrow.parquet as pq
import logging
from tqdm import tqdm

from constants import ProcessingConstants, ParquetConstants
from deduplication_processor import DeduplicationProcessor
from optimized_deduplication import OptimizedDeduplicationProcessor
from logger_config import LoggerConfig

# Debug configuration
DEBUG_CONFIG = {
    'model': 'Mistral-7B-Instruct-v0.3',
    'language': 'en',
    'dataset': 'mmlu.logical_fallacies',
    'shots': '5_shot'
}

class DebugDatasetMerger:
    def __init__(
            self,
            data_dir: Path,
            num_workers: int = ProcessingConstants.DEFAULT_NUM_WORKERS
    ):
        self.source_dir = data_dir
        # Create debug directory by appending '_debug' to the original path
        self.debug_dir = Path(str(data_dir) + '_debug')
        self.num_workers = num_workers
        
        # Setup logger
        self.logger = LoggerConfig.setup_logger(
            "DebugDatasetMerger", 
            self.debug_dir / ProcessingConstants.LOGS_DIR_NAME
        )

    def debug_process(self):
        """Run debug processing pipeline for specific dataset."""
        try:
            self.logger.info(f"Starting debug process for {DEBUG_CONFIG}")
            self.logger.info(f"Source directory: {self.source_dir}")
            self.logger.info(f"Debug directory: {self.debug_dir}")

            # 1. Create debug directory structure and copy relevant files
            self.logger.info("Setting up debug environment...")
            if not self._setup_debug_environment():
                return

            # 2. Process the copied files
            debug_dataset_dir = self.debug_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME / \
                              DEBUG_CONFIG['model'] / DEBUG_CONFIG['language'] / \
                              DEBUG_CONFIG['shots'] / DEBUG_CONFIG['dataset']

            # Count rows in copied files
            total_rows = 0
            file_counts = {}
            self.logger.info("Counting rows in original files:")
            for file in tqdm(debug_dataset_dir.glob("*.parquet"), desc="Counting files"):
                pf = pq.ParquetFile(str(file))
                rows = pf.metadata.num_rows
                total_rows += rows
                file_counts[file.name] = rows
                self.logger.info(f"  {file.name}: {rows:,} rows")
            self.logger.info(f"Total rows before merge: {total_rows:,}")

            # 3. Merge files
            self.logger.info("\nStarting merge process...")
            merged_file = self._merge_dataset_files(
                self.debug_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME / DEBUG_CONFIG['model'],
                Path(DEBUG_CONFIG['language']),
                Path(DEBUG_CONFIG['shots']),
                debug_dataset_dir
            )

            # Count rows in merged file
            if merged_file and merged_file.exists():
                merged_rows = pq.ParquetFile(str(merged_file)).metadata.num_rows
                self.logger.info(f"Merged file rows: {merged_rows:,}")
                if merged_rows != total_rows:
                    self.logger.warning(
                        f"Row count mismatch! Original: {total_rows:,}, Merged: {merged_rows:,}"
                    )
            else:
                self.logger.error("Error merging files")
                raise Exception("Error merging files")

            # 4. Deduplicate
            self.logger.info("\nStarting deduplication...")
            deduplicator = OptimizedDeduplicationProcessor(self.debug_dir)
            deduplicator.deduplicate_files({str(merged_file)})

            # Count rows after deduplication
            if merged_file.exists():
                final_rows = pq.ParquetFile(str(merged_file)).metadata.num_rows
                self.logger.info(f"Final results:")
                self.logger.info(f"  Original rows: {total_rows:,}")
                self.logger.info(f"  After merge: {merged_rows:,}")
                self.logger.info(f"  After deduplication: {final_rows:,}")
                self.logger.info(f"  Duplicates removed: {merged_rows - final_rows:,}")

        except Exception as e:
            self.logger.error(f"Error in debug process: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _setup_debug_environment(self):
        """Setup debug environment by copying relevant files."""
        try:
            # Create debug directory structure
            debug_full_schema_dir = self.debug_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME
            debug_full_schema_dir.mkdir(parents=True, exist_ok=True)

            # Source directory for specific dataset
            source_dataset_dir = self.source_dir / ProcessingConstants.FULL_SCHEMA_DIR_NAME / \
                               DEBUG_CONFIG['model'] / DEBUG_CONFIG['language'] / \
                               DEBUG_CONFIG['shots'] / DEBUG_CONFIG['dataset']

            if not source_dataset_dir.exists():
                self.logger.error(f"Source directory not found: {source_dataset_dir}")
                return False

            # Copy dataset directory structure and files
            debug_dataset_dir = debug_full_schema_dir / DEBUG_CONFIG['model'] / \
                              DEBUG_CONFIG['language'] / DEBUG_CONFIG['shots'] / \
                              DEBUG_CONFIG['dataset']
            
            # Remove existing debug directory if it exists
            if debug_dataset_dir.exists():
                shutil.rmtree(debug_dataset_dir)
            
            # Copy the directory
            shutil.copytree(source_dataset_dir, debug_dataset_dir)
            self.logger.info(f"Copied dataset files to: {debug_dataset_dir}")
            
            return True

        except Exception as e:
            self.logger.error(f"Error setting up debug environment: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _merge_dataset_files(self, model_dir: Path, lang_dir: Path, shots_dir: Path, dataset_dir: Path) -> Path:
        """Merge all files in a dataset directory into a single parquet file."""
        try:
            parquet_files = list(dataset_dir.glob(f"*{ProcessingConstants.PARQUET_EXTENSION}"))
            if not parquet_files:
                return None

            merged_path = model_dir / lang_dir.name / shots_dir.name / f"{dataset_dir.name}.parquet"
            merged_path.parent.mkdir(parents=True, exist_ok=True)

            writer = None
            for input_file in parquet_files:
                pf = pq.ParquetFile(str(input_file))
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

            # Clean up individual files and directory
            for f in parquet_files:
                f.unlink()
            dataset_dir.rmdir()

            return merged_path

        except Exception as e:
            self.logger.error(f"Error merging files in {dataset_dir}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

if __name__ == "__main__":
    debug_merger = DebugDatasetMerger(
        data_dir=Path(ProcessingConstants.OUTPUT_DATA_DIR),
        num_workers=1  # Single worker for debugging
    )
    debug_merger.debug_process() 