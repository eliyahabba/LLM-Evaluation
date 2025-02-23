import polars as pl
from pathlib import Path
from filelock import FileLock
from logger_config import LoggerConfig
from constants import ProcessingConstants

class OptimizedDeduplicationProcessor:
    """Memory-efficient deduplication processor for large parquet files."""
    
    def __init__(self, data_dir: Path):
        """Initialize the optimized deduplication processor."""
        self.data_dir = data_dir
        self.logger = LoggerConfig.setup_logger(
            "OptimizedDeduplicationProcessor", 
            self.data_dir / ProcessingConstants.LOGS_DIR_NAME
        )

        # Define deduplication expressions for nested fields
        self.dedup_expressions = [
            pl.col('instance').struct.field('sample_identifier').struct.field('hf_index').alias('sample_index'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('shots').alias('shots'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('choices_order').struct.field('method').alias('choices_order_method'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('enumerator').alias('enumerator'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('separator').alias('separator'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('instruction_phrasing').struct.field('name').alias('instruction_phrasing_name')
        ]
        
        # Define deduplication keys (removed )
        self.dedup_keys = [
            'sample_index', 'shots', 'choices_order_method',
            'enumerator', 'separator', 'instruction_phrasing_name'
        ]

    def deduplicate_files(self, file_paths: set) -> bool:
        """
        Deduplicate data in parquet files using memory-efficient lazy evaluation.
        
        Args:
            file_paths: Set of file paths to deduplicate
            
        Returns:
            bool: True if deduplication was successful
        """
        try:
            for file_path in file_paths:
                try:
                    self.logger.info(f"Deduplicating {file_path}")
                    path = Path(file_path)
                    
                    if not path.exists():
                        self.logger.warning(f"File not found: {file_path}")
                        continue

                    # Get original row count
                    original_count = pl.scan_parquet(str(path)).select(pl.count()).collect().item()
                    # Read the entire file and then convert to lazy
                    df_lazy = pl.read_parquet(str(path)).lazy().select(["instance", "prompt_config", "raw_input"])

                    # Perform deduplication
                    deduped_df = self._deduplicate_file(path)
                    
                    # Log results
                    duplicates_removed = original_count - len(deduped_df)
                    self.logger.info(
                        f"Deduplication results for {path.name}: "
                        f"removed {duplicates_removed} duplicates, "
                        f"{len(deduped_df)} rows remaining"
                    )

                except Exception as e:
                    self.logger.error(f"Error deduplicating {file_path}: {e}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error in deduplicate_files: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _deduplicate_file(self, path: Path) -> pl.DataFrame:
        """
        Deduplicate a single parquet file using lazy evaluation.
        
        Args:
            path: Path to the parquet file
            
        Returns:
            Deduplicated DataFrame
        """
        lock_file = path.parent / f"{path.stem}.lock"
        temp_file = path.parent / f"{path.stem}_dedup.parquet"
        
        with FileLock(str(lock_file)):
            # Get original row count
            original_count = pl.scan_parquet(str(path)).select(pl.count()).collect().item()
            
            # Step 1: Lazy read only necessary columns for deduplication and filtering
            # dedup_lazy = pl.scan_parquet(
            #     str(path),
            #     columns=["instance", "prompt_config"]
            # ).with_row_count("_row_idx")
            dedup_lazy = pl.scan_parquet(str(path)).select(["instance", "prompt_config"]).with_row_count("_row_idx")

            # Add columns for filtering and deduplication
            dedup_lazy = dedup_lazy.select(
                pl.col("_row_idx"),
                *self.dedup_expressions  # Remove raw_input from selection
            )
            
            # Filter out unwanted rows
            filtered_lazy = dedup_lazy.filter(
                ~(
                    (pl.col("shots") == 5) & 
                    pl.col("choices_order_method").is_in(["correct_first", "correct_last"])
                )
            )
            
            # Count filtered rows
            filtered_count = filtered_lazy.select(pl.count()).collect().item()
            removed_by_filter = original_count - filtered_count
            self.logger.info(f"Removed {removed_by_filter:,} rows by filtering (shots=5 and correct_first/last)")
            
            # Get unique row indices
            unique_indices = filtered_lazy.unique(
                subset=self.dedup_keys,
                maintain_order=True
            ).select("_row_idx").collect()["_row_idx"].to_list()
            
            # Step 2: Filter full file to keep only unique rows
            dedup_full_lazy = pl.scan_parquet(str(path)) \
                .with_row_count("_row_idx") \
                .filter(pl.col("_row_idx").is_in(unique_indices)) \
                .drop("_row_idx")
            
            # Collect results and save
            df_dedup = dedup_full_lazy.collect()
            final_count = len(df_dedup)
            
            # Log deduplication results
            duplicates_removed = filtered_count - final_count
            self.logger.info(f"Deduplication results for {path.name}:")
            self.logger.info(f"  Original rows: {original_count:,}")
            self.logger.info(f"  After filtering: {filtered_count:,}")
            self.logger.info(f"  After deduplication: {final_count:,}")
            self.logger.info(f"  Rows removed by filtering: {removed_by_filter:,}")
            self.logger.info(f"  Duplicates removed: {duplicates_removed:,}")
            self.logger.info(f"  Total rows removed: {original_count - final_count:,}")
            
            # Save results
            df_dedup.write_parquet(str(temp_file))
            temp_file.replace(path)
            
            return df_dedup 