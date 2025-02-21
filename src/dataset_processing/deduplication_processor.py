# deduplication_processor.py
from pathlib import Path
import polars as pl
from base_processor import BaseProcessor
from filelock import FileLock
from typing import Set
from constants import ProcessingConstants
from logger_config import LoggerConfig


class DeduplicationProcessor:
    def __init__(self, data_dir: str):
        """
        Initialize the deduplication processor.
        
        Args:
            data_dir: Directory containing files to deduplicate
        """
        self.data_dir = Path(data_dir)
        
        # Setup logger
        log_dir = self.data_dir / ProcessingConstants.LOGS_DIR_NAME
        self.logger = LoggerConfig.setup_logger("DeduplicationProcessor", log_dir)
        
        # Define deduplication columns for full schema
        self.full_schema_expressions = [
            pl.col('model').struct.field('model_info').struct.field('name').alias('_model_name'),
            pl.col('instance').struct.field('task_type').alias('_task_type'),
            pl.col('instance').struct.field('classification_fields').struct.field('question').alias('_question'),
            pl.col('instance').struct.field('classification_fields').struct.field('choices').alias('_choices'),
            pl.col('instance').struct.field('classification_fields').struct.field('ground_truth').alias('_ground_truth'),
            pl.col('instance').struct.field('sample_identifier').struct.field('dataset_name').alias('_dataset_name')
        ]
        
        self.full_schema_dedup_cols = [
            '_model_name', '_task_type', '_question', '_choices', 
            '_ground_truth', '_dataset_name'
        ]
        
        # Define deduplication columns for lean schema
        self.lean_schema_dedup_cols = [
            'model', 'dataset', 'sample_index', 'question','shots','instruction_phrasing_name', 'separator', 'enumerator', 'choices_order'
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
                        
                        # Determine if this is a full or lean schema file
                        is_full_schema = 'model' in df.columns and isinstance(df['model'][0], dict)
                        
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