import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

# Configure basic logger
logger = logging.getLogger("dedup_logger")
logger.setLevel(logging.INFO)

# Add StreamHandler (to display logs on the console) if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def process_parquet_file(parquet_file: Path, input_path: Path, all_cols_dedup_dir: Path):
    # def process_parquet_file(parquet_file: Path, input_path: Path, dedup_eval_id_dir: Path, all_cols_dedup_dir: Path):
    """
    Processes a single Parquet file:
    1. Reads the file.
    2. Creates two deduplicated versions:
       a. Deduplication based on 'evaluation_id'.
       b. Deduplication based on all columns (excluding 'index' if present).
    3. Saves both versions in corresponding output directories.
    4. Returns processing metadata (row count, duplicate count) for logging.
    """
    try:
        logger.info(f"[START] - Processing file: {parquet_file}")

        # Read the Parquet file
        df = pl.read_parquet(str(parquet_file))
        logger.info(f"  Read {df.shape[0]} rows and {df.shape[1]} columns from {parquet_file}")
        df = df.with_row_count("_row_idx")

        # Create expressions for extracting nested fields efficiently
        nested_expressions = [
            pl.col('model').struct.field('model_info').struct.field('name').alias('_model_name'),
            pl.col('prompt_config').struct.field('format').struct.field('shots').alias('_shots'),
            pl.col('instance').struct.field('sample_identifier').struct.field('dataset_name').alias('_dataset_name'),
            pl.col('instance').struct.field('sample_identifier').struct.field('hf_index').alias('_dataset_index'),
            pl.col('prompt_config').struct.field('format').struct.field('choices_order').struct.field('method').alias('_choices_order'),
            pl.col('prompt_config').struct.field('format').struct.field('enumerator').alias('_enumerator'),
            pl.col('prompt_config').struct.field('format').struct.field('separator').alias('_separator'),
            pl.col('prompt_config').struct.field('format').struct.field('template').alias('_template')
        ]

        # Add temporary columns for deduplication
        df_with_fields = df.with_columns(nested_expressions)

        # Define deduplication columns (temporary columns)
        dedup_cols = [
            '_model_name', '_shots', '_dataset_name', '_dataset_index',
            '_choices_order', '_enumerator', '_separator', '_template'
        ]

        # Get metrics before deduplication
        total_rows = df_with_fields.shape[0]

        # Get unique row indices based on the temporary columns
        unique_indices = df_with_fields.unique(subset=dedup_cols, maintain_order=True).get_column('_row_idx')
        # Filter original dataframe using these indices
        df_dedup = df.filter(pl.col('_row_idx').is_in(unique_indices))

        distinct_rows = df_dedup.shape[0]
        duplicates_count = total_rows - distinct_rows

        logger.info(f"  Deduplication by nested fields: "
                    f"Found {duplicates_count} duplicates, "
                    f"{distinct_rows} rows remaining")

        # Save deduplicated file (with original structure)
        all_cols_file_path = all_cols_dedup_dir / parquet_file.relative_to(input_path)
        all_cols_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_dedup.write_parquet(str(all_cols_file_path))
        logger.info(f"  Saved deduplicated file at: {all_cols_file_path}")

        logger.info(f"[DONE] - Finished processing: {parquet_file}\n")
        return parquet_file, total_rows, duplicates_count

    except Exception as e:
        logger.error(f"[ERROR] Error processing file {parquet_file}: {e}", exc_info=True)
        return parquet_file, None, None, str(e)


def process_parquet_files_parallel(input_dir: str, max_workers: int = 4):
    """
    1. Creates two output directories (mirroring the input structure):
       - <input_dir>_eval_id_deduped
       - <input_dir>_all_cols_deduped
    2. Recursively searches for Parquet files.
    3. Processes them in parallel using ThreadPoolExecutor.
    """
    input_path = Path(input_dir).resolve()

    # Define output directories
    # dedup_eval_id_dir = input_path.parent / (input_path.name + "_eval_id_deduped")
    all_cols_dedup_dir = input_path.parent / (input_path.name + "_all_cols_deduped")
    # dedup_eval_id_dir.mkdir(parents=True, exist_ok=True)
    all_cols_dedup_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_path.rglob("*.parquet"))
    if not parquet_files:
        logger.warning(f"[WARNING] No .parquet files found under {input_dir}")
        return

    logger.info(f"Found {len(parquet_files)} Parquet files in directory {input_dir}")
    logger.info(f"Starting parallel processing with max_workers={max_workers}\n")

    # Parallel processing of the files
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_parquet_file, pf, input_path, all_cols_dedup_dir)
            for pf in parquet_files
        ]
        # futures = [
        #     executor.submit(process_parquet_file, pf, input_path, dedup_eval_id_dir, all_cols_dedup_dir)
        #     for pf in parquet_files
        # ]

        for future in as_completed(futures):
            results.append(future.result())

    # Print processing summary
    logger.info("=== Processing Summary ===")
    for item in results:
        if len(item) == 4 and item[3] is not None:
            # Error occurred
            logger.error(f"[FAILED] File: {item[0]}, Error: {item[3]}")
        else:
            parquet_file, total_rows, duplicates_count_all_cols = item
            logger.info(
                f"[OK] - {parquet_file}\n"
                f"       Total rows: {total_rows}, Duplicates by all columns: {duplicates_count_all_cols}"
            )

    logger.info("\nDone processing all Parquet files in parallel!\n")


if __name__ == "__main__":
    input_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_split"
    # first if there is tmp folder, remove it
    tmp_dir = Path(input_dir) / "temp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    process_parquet_files_parallel(input_dir, max_workers=24)
    # parquet_file = pq.ParquetFile(input_file)
    # for batch in parquet_file.iter_batches(batch_size=10):
    #     pass
