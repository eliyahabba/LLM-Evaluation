# deduplication_processor.py
from pathlib import Path
import polars as pl
from base_processor import BaseProcessor
from filelock import FileLock
from typing import Set


class DeduplicationProcessor(BaseProcessor):
    def __init__(self, **kwargs):
        """Initialize the deduplication processor."""
        super().__init__(**kwargs)
        
        # Define nested expressions for deduplication
        self.nested_expressions = [
            pl.col('model').struct.field('model_info').struct.field('name').alias('_model_name'),
            pl.col('instance').struct.field('task_type').alias('_task_type'),
            pl.col('instance').struct.field('classification_fields').struct.field('question').alias('_question'),
            pl.col('instance').struct.field('classification_fields').struct.field('choices').alias('_choices'),
            pl.col('instance').struct.field('classification_fields').struct.field('ground_truth').alias('_ground_truth'),
            pl.col('instance').struct.field('sample_identifier').struct.field('dataset_name').alias('_dataset_name')
        ]
        
        self.dedup_cols = [
            '_model_name', '_task_type', '_question', '_choices', 
            '_ground_truth', '_dataset_name'
        ]

    def deduplicate_all_files(self) -> bool:
        """
        Deduplicate all files in the data directory structure.
        
        Returns:
            bool: True if all deduplication was successful
        """
        try:
            # Get all parquet files in the directory structure
            parquet_files = list(self.data_dir.rglob("*.parquet"))
            if not parquet_files:
                self.logger.warning("No files found for deduplication")
                return False

            success = True
            for file_path in parquet_files:
                try:
                    if not file_path.exists():
                        self.logger.warning(f"File not found for deduplication: {file_path}")
                        continue

                    lock_file = file_path.parent / f"{file_path.stem}.lock"
                    temp_file = file_path.parent / f"{file_path.stem}_dedup.parquet"

                    with FileLock(lock_file):
                        # Read the Parquet file
                        df = pl.read_parquet(str(file_path))
                        self.logger.info(f"Read {df.shape[0]} rows from {file_path}")
                        df = df.with_row_count("_row_idx")

                        # Add temporary columns for deduplication
                        df_with_fields = df.with_columns(self.nested_expressions)

                        # Get metrics before deduplication
                        total_rows = df_with_fields.shape[0]

                        # Get unique row indices based on the temporary columns
                        unique_indices = df_with_fields.unique(subset=self.dedup_cols, maintain_order=True).get_column('_row_idx')
                        # Filter original dataframe using these indices
                        df_dedup = df.filter(pl.col('_row_idx').is_in(unique_indices))

                        distinct_rows = df_dedup.shape[0]
                        duplicates_count = total_rows - distinct_rows

                        self.logger.info(
                            f"Deduplication results for {file_path.name}: "
                            f"removed {duplicates_count} duplicates, "
                            f"{distinct_rows} rows remaining"
                        )

                        # Write deduplicated data to temporary file
                        df_dedup.write_parquet(str(temp_file))

                        # Replace original file with deduplicated version
                        temp_file.replace(file_path)

                except Exception as e:
                    self.logger.error(f"Error deduplicating {file_path}: {e}")
                    if temp_file.exists():
                        temp_file.unlink()
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Error in deduplicate_all_files: {e}")
            return False

    def deduplicate_files(self, file_paths: Set[str]) -> bool:
        """
        Deduplicate specific files from this run.
        
        Args:
            file_paths: Set of file paths to deduplicate
            
        Returns:
            bool: True if deduplication was successful
        """
        try:
            if not file_paths:
                self.logger.warning("No files to deduplicate")
                return False

            success = True
            for file_path in file_paths:
                try:
                    if not Path(file_path).exists():
                        self.logger.warning(f"File not found for deduplication: {file_path}")
                        continue

                    # ... rest of deduplication logic ...

                except Exception as e:
                    self.logger.error(f"Error deduplicating {file_path}: {e}")
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Error in deduplicate_files: {e}")
            return False