# deduplication_processor.py
from pathlib import Path
from typing import Set

import polars as pl
from filelock import FileLock

from constants import ProcessingConstants
from logger_config import LoggerConfig


class DeduplicationProcessor:
    def __init__(self, data_dir: Path, schema_type: str):
        """Initialize the deduplication processor."""
        self.data_dir = data_dir
        self.schema_type = schema_type

        # Setup logger
        log_dir = self.data_dir / ProcessingConstants.LOGS_DIR_NAME
        self.logger = LoggerConfig.setup_logger("DeduplicationProcessor", log_dir)

        # Full schema deduplication expressions
        self.full_schema_expressions = [
            pl.col('instance').struct.field('sample_identifier').struct.field('hf_index').alias('sample_index'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('shots').alias('shots'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('choices_order').struct.field(
                'method').alias('choices_order_method'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('enumerator').alias('enumerator'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('separator').alias('separator'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('instruction_phrasing').struct.field(
                'name').alias('instruction_phrasing_name'),
        ]

        # Fields to check for duplicates in full schema
        self.full_schema_dedup_cols = [
            'sample_index', 'raw_input',
            'shots', 'choices_order_method',
            'enumerator', 'separator', 'instruction_phrasing_name',
        ]

        # Fields to check for duplicates in lean schema
        self.lean_schema_dedup_cols = [
            'sample_index', 'shots', 'choices_order',
            'enumerator', 'separator', 'instruction_phrasing',
        ]

    def deduplicate_files(self, file_paths: Set[str]) -> bool:
        """
        Deduplicate each file from the current run.
        
        Args:
            file_paths: Set of file paths to deduplicate
            
        Returns:
            bool: True if all deduplication was successful
        """
        try:
            if not file_paths:
                self.logger.warning("No files to deduplicate")
                return False

            success = True
            for file_path in file_paths:
                try:
                    path = Path(file_path)
                    if not path.exists():
                        self.logger.warning(f"File not found for deduplication: {file_path}")
                        continue

                    lock_file = path.parent / f"{path.stem}.lock"
                    temp_file = path.parent / f"{path.stem}_dedup.parquet"

                    with FileLock(lock_file):
                        # Read the Parquet file
                        df = pl.read_parquet(str(path))
                        self.logger.info(f"Read {df.shape[0]} rows from {path}")

                        # Use schema type from initialization
                        is_full_schema = self.schema_type == "full"

                        if is_full_schema:
                            # Full schema deduplication
                            df = df.with_row_count("_row_idx")
                            df_with_fields = df.with_columns(self.full_schema_expressions)
                            unique_indices = df_with_fields.unique(
                                subset=self.full_schema_dedup_cols,
                                maintain_order=True
                            ).get_column('_row_idx')
                            df_dedup = df.filter(pl.col('_row_idx').is_in(unique_indices))
                        else:
                            # Lean schema deduplication
                            df_dedup = df.unique(subset=self.lean_schema_dedup_cols, maintain_order=True)

                        # Log deduplication results
                        duplicates_removed = df.shape[0] - df_dedup.shape[0]
                        self.logger.info(
                            f"Deduplication results for {path.name}: "
                            f"removed {duplicates_removed} duplicates, "
                            f"{df_dedup.shape[0]} rows remaining"
                        )

                        # Write deduplicated data
                        df_dedup.write_parquet(str(temp_file))
                        temp_file.replace(path)

                except Exception as e:
                    self.logger.error(f"Error deduplicating {file_path}: {e}")
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Error in deduplicate_files: {e}")
            return False
