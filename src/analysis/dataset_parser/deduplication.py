import logging
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl


def setup_logger(name="dedup_logger"):
    """Configure and return a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def process_parquet_file(parquet_file: Path, input_path: Path, all_cols_dedup_dir: Path):
    """
    Process a single Parquet file for deduplication.

    Args:
        parquet_file: Path to the input parquet file
        input_path: Base input directory path
        all_cols_dedup_dir: Output directory for deduplicated files

    Returns:
        Tuple containing file path, total rows, and duplicate count
    """
    logger = logging.getLogger("dedup_logger")

    try:
        logger.info(f"[START] - Processing file: {parquet_file}")

        # Read the Parquet file
        df = pl.read_parquet(str(parquet_file))
        logger.info(f"  Read {df.shape[0]} rows and {df.shape[1]} columns from {parquet_file}")

        # Deduplicate based on all columns (excluding 'index' if present)
        exclude_cols = ["index"] if "index" in df.columns else []
        all_cols_except_index = [c for c in df.columns if c not in exclude_cols]

        total_rows = df.shape[0]
        distinct_by_all = df.unique(subset=all_cols_except_index).shape[0]
        duplicates_count_all_cols = total_rows - distinct_by_all
        df_all_cols_dedup = df.unique(subset=all_cols_except_index)

        logger.info(f"  Deduplication by all columns (excluding {exclude_cols}): "
                    f"Found {duplicates_count_all_cols} duplicates, "
                    f"{df_all_cols_dedup.shape[0]} rows remaining")

        # Save results in a mirrored directory structure
        relative_path = parquet_file.relative_to(input_path)
        all_cols_file_path = all_cols_dedup_dir / relative_path
        all_cols_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_all_cols_dedup.write_parquet(str(all_cols_file_path))
        logger.info(f"  Saved deduplicated file at: {all_cols_file_path}")

        logger.info(f"[DONE] - Finished processing: {parquet_file}\n")
        return parquet_file, total_rows, duplicates_count_all_cols

    except Exception as e:
        logger.error(f"[ERROR] Error processing file {parquet_file}: {e}", exc_info=True)
        return parquet_file, None, None, str(e)


def process_parquet_files_parallel(input_dir: str, max_workers: int = 4):
    """
    Process multiple parquet files in parallel for deduplication.

    Args:
        input_dir: Input directory containing parquet files
        max_workers: Number of parallel workers
    """
    logger = logging.getLogger("dedup_logger")
    input_path = Path(input_dir).resolve()

    # Define output directory
    all_cols_dedup_dir = input_path.parent / (input_path.name + "_all_cols_deduped")
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


def main():
    parser = argparse.ArgumentParser(description='Process and deduplicate Parquet files in parallel.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing Parquet files')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads (default: 4)')
    parser.add_argument('--clean-temp', action='store_true',
                        help='Clean temporary directory before processing')

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # Clean temp directory if requested
    if args.clean_temp:
        tmp_dir = Path(args.input_dir) / "temp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            logger.info(f"Cleaned temporary directory: {tmp_dir}")

    # Process files
    process_parquet_files_parallel(args.input_dir, max_workers=args.workers)


if __name__ == "__main__":
    main()