import re
import shutil
from pathlib import Path


def reorganize_files(input_dir, output_dir):
    """
    Reads all files in the format: <model>_shots<shots>_<dataset>.parquet
    and moves them into the following directory structure:
        output_dir/safe_model/shots_<shots>/english/<dataset>.parquet
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all .parquet files in the input directory (without subdirectories)
    for file_path in input_dir.glob("*.parquet"):
        # Try to extract different parts from the filename
        match = re.match(r'(?P<model>.+)_shots(?P<shots>\d+)_(?P<dataset>.+)\.parquet', file_path.name)
        if not match:
            # If the pattern does not match, log a message and continue
            print(f"Unrecognized format for file: {file_path.name}")
            continue

        safe_model = match.group('model')  # Model name (everything before "_shotsX")
        shots = match.group('shots')  # Number of shots
        dataset = match.group('dataset')  # Dataset name (everything after "shotsX_" until the extension)

        # Define the target directory structure
        lang_match = re.match(r'(global_mmlu|global_mmlu_lite_cs|global_mmlu_lite_ca)\.(\w+)$', dataset)

        if lang_match:
            language = lang_match.group(2).lower()
        else:
            language = "english"


        target_dir = output_dir / safe_model / f"shots_{shots}" / language
        target_dir.mkdir(parents=True, exist_ok=True)

        # Define the new file path
        new_path = target_dir / f"{dataset}.parquet"

        # Move (or copy) the file to the new location
        print(f"Moving {file_path.name} to {new_path}")
        shutil.copy(str(file_path), str(new_path))  # Use shutil.copy if copying is preferred

    print("File reorganization complete!")


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split"
    output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split_to_folders"

    reorganize_files(input_dir, output_dir)