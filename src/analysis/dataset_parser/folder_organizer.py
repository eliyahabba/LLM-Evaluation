import re
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


def process_single_file(file_path, output_dir):
    """
    Process a single parquet file and organize it into the target directory structure.

    Args:
        file_path: Path to the input parquet file
        output_dir: Base output directory

    Returns:
        str: Status message about the processing
    """
    match = re.match(r'(?P<model>.+)_shots(?P<shots>\d+)_(?P<dataset>.+)\.parquet', file_path.name)
    if not match:
        return f"Unrecognized format for file: {file_path.name}"

    safe_model = match.group('model')
    shots = match.group('shots')
    dataset = match.group('dataset')

    # Handle language-specific datasets
    lang_match = re.match(r'(global_mmlu|global_mmlu_lite_cs|global_mmlu_lite_ca)\.(\w+)$', dataset)
    language = lang_match.group(2).lower() if lang_match else "en"

    # Create target directory structure
    target_dir = output_dir / safe_model / f"shots_{shots}" / language
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy file to new location
    new_path = target_dir / f"{dataset}.parquet"
    shutil.copy(str(file_path), str(new_path))

    return f"Processed {file_path.name}"


def reorganize_files(input_dir: str, output_dir: str, num_workers: int = 12):
    """
    Reorganize parquet files into a structured directory layout.

    Args:
        input_dir: Directory containing input parquet files
        output_dir: Base directory for reorganized files
        num_workers: Number of parallel workers
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of parquet files
    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files to process")

    # Process files in parallel with progress bar
    results = []
    with tqdm(total=len(parquet_files), desc="Processing files") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_single_file, pf, output_dir)
                for pf in parquet_files
            ]

            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    # Print summary
    print("\nProcessing Summary:")
    success_count = sum(1 for r in results if not r.startswith("Unrecognized"))
    failed_count = len(results) - success_count
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {failed_count} files")

    if failed_count > 0:
        print("\nFailed files:")
        for result in results:
            if result.startswith("Unrecognized"):
                print(result)

    print("\nFile reorganization complete!")


def get_output_dir(input_dir: str) -> str:
    """
    Generate output directory name based on input directory pattern.
    Converts patterns like:
    - '*_all_cols_deduped' -> '*_to_folders'
    - '*_processed_split' -> '*_split_to_folders'
    """
    input_path = Path(input_dir)
    if input_path.name.endswith('_all_cols_deduped'):
        base_name = input_path.name[:-len('_all_cols_deduped')]
        return str(input_path.parent / f"{base_name}_to_folders")
    elif input_path.name.endswith('_processed_split'):
        base_name = input_path.name[:-len('_processed_split')]
        return str(input_path.parent / f"{base_name}_processed_split_to_folders")
    else:
        return str(input_path.parent / f"{input_path.name}_to_folders")


def main():
    parser = argparse.ArgumentParser(description='Reorganize parquet files into structured directories.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing parquet files')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of worker threads (default: 12)')

    args = parser.parse_args()

    output_dir = get_output_dir(args.input_dir)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")

    reorganize_files(args.input_dir, output_dir, args.workers)


if __name__ == "__main__":
    main()