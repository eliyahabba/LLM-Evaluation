import polars as pl
from pathlib import Path
from filelock import FileLock
from tqdm import tqdm

from logger_config import LoggerConfig
from constants import ProcessingConstants
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


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
            pl.col('prompt_config').struct.field('dimensions').struct.field('choices_order').struct.field(
                'method').alias('choices_order_method'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('enumerator').alias('enumerator'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('separator').alias('separator'),
            pl.col('prompt_config').struct.field('dimensions').struct.field('instruction_phrasing').struct.field(
                'name').alias('instruction_phrasing_name')
        ]

        self.dedup_keys = [
            'sample_index', 'shots', 'choices_order_method',
            'enumerator', 'separator', 'instruction_phrasing_name'
        ]

    def deduplicate_files(self, merged_file_paths: set) -> bool:
        """
        Deduplicate data in parquet files using memory-efficient processing.

        Args:
            merged_file_paths: Set of file paths to deduplicate

        Returns:
            bool: True if deduplication was successful
        """
        try:
            for merged_file_path in merged_file_paths:
                try:
                    self.logger.info(f"Deduplicating {merged_file_path}")
                    output_path = Path(merged_file_path)
                    # remove the "merged" suffix from the file name
                    output_path = output_path.parent / output_path.name.replace("_merged", "")

                    if not merged_file_path.exists():
                        self.logger.warning(f"File not found: {merged_file_path}")
                        continue

                    # Perform deduplication
                    deduped_df = self._deduplicate_file(merged_file_path, output_path)

                    # Log results
                    original_count = pl.scan_parquet(str(merged_file_path)).select(pl.count()).collect().item()
                    duplicates_removed = original_count - len(deduped_df)
                    self.logger.info(
                        f"Deduplication results for {merged_file_path.name}: "
                        f"removed {duplicates_removed} duplicates, "
                        f"{len(deduped_df)} rows remaining"
                    )

                except Exception as e:
                    self.logger.error(f"Error deduplicating {merged_file_path}: {e}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error in deduplicate_files: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _deduplicate_file(self, merged_path: Path, output_path: Path) -> pl.DataFrame:
        """
        Deduplicate a single parquet file using lazy evaluation for filtering
        and PyArrow for batch writing to avoid high memory usage.

        Args:
            merged_path: Path to the parquet file

        Returns:
            Deduplicated DataFrame (loaded after writing the deduplicated file)
        """

        if output_path.exists():
            self.logger.info(f"Output file {output_path} exists. Merging it with new data from {merged_path}.")
            combined_file = merged_path.parent / f"{merged_path.stem}_combined.parquet"
            writer = None
            for file in [output_path, merged_path]:
                pf = pq.ParquetFile(str(file))
                total_row_counter = 0
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg)
                    num_rows = table.num_rows
                    row_indices = pa.array(range(total_row_counter, total_row_counter + num_rows))
                    table = table.append_column("_row_idx", row_indices)
                    total_row_counter += num_rows
                    if writer is None:
                        writer = pq.ParquetWriter(str(combined_file), table.schema)
                    writer.write_table(table.drop(["_row_idx"]))
            if writer is not None:
                writer.close()
            output_path.unlink()
            merged_path.unlink()
            merged_path = combined_file

        temp_file = merged_path.parent / f"{merged_path.stem}_dedup.parquet"

        # Get original row count
        original_count = pl.scan_parquet(str(merged_path)).select(pl.count()).collect().item()

        # Step 1: Lazy read only necessary columns for deduplication and filtering
        dedup_lazy = pl.scan_parquet(str(merged_path)).select(["instance", "prompt_config"]).with_row_count("_row_idx")

        # Add columns for filtering and deduplication
        dedup_lazy = dedup_lazy.select(
            pl.col("_row_idx"),
            *self.dedup_expressions  # Remove raw_input from selection if קיימת
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

        # Step 2: Extract unique indices based on deduplication keys
        unique_indices_lazy = filtered_lazy.select(["_row_idx"] + self.dedup_keys).unique(
            subset=self.dedup_keys,
            maintain_order=True
        ).select("_row_idx")
        unique_indices_df = unique_indices_lazy.collect()
        self.logger.info(f"Unique indices count: {unique_indices_df.shape[0]}")

        # Convert unique indices to a set for fast membership checking in PyArrow
        unique_indices_set = set(unique_indices_df.to_pandas()['_row_idx'])
        self.logger.info(f"Unique indices set length: {len(unique_indices_set)}")

        # Step 3: Process the original parquet file in batches using PyArrow
        batch_size = 1000
        pf = pq.ParquetFile(str(merged_path))
        writer = None
        total_row_counter = 0

        for rg in tqdm(range(pf.num_row_groups)):
            table = pf.read_row_group(rg)
            num_rows = table.num_rows

            row_indices = pa.array(range(total_row_counter, total_row_counter + num_rows))
            table = table.append_column("_row_idx", row_indices)
            total_row_counter += num_rows

            unique_indices_array = pa.array(list(unique_indices_set))
            mask = pc.is_in(table["_row_idx"], value_set=unique_indices_array)
            filtered_table = table.filter(mask)

            num_filtered = filtered_table.num_rows
            if num_filtered > 0:
                for start in range(0, num_filtered, batch_size):
                    batch = filtered_table.slice(start, batch_size)
                    batch = batch.drop(["_row_idx"])
                    if writer is None:
                        writer = pq.ParquetWriter(str(temp_file), batch.schema)
                    writer.write_table(batch)

            if writer is not None:
                writer.close()

            # Optionally, replace the original file with the deduplicated version
            temp_file.replace(output_path)
            #  unlink the merged file
            merged_path.unlink()

            # Load the deduplicated file into a Polars DataFrame and return
            df_dedup = pl.read_parquet(str(merged_path))
            final_count = len(df_dedup)
            duplicates_removed = filtered_count - final_count
            self.logger.info(f"Deduplication results for {merged_path.name}:")
            self.logger.info(f"  Original rows: {original_count:,}")
            self.logger.info(f"  After filtering: {filtered_count:,}")
            self.logger.info(f"  After deduplication: {final_count:,}")
            self.logger.info(f"  Rows removed by filtering: {removed_by_filter:,}")
            self.logger.info(f"  Duplicates removed: {duplicates_removed:,}")
            self.logger.info(f"  Total rows removed: {original_count - final_count:,}")

            return df_dedup
