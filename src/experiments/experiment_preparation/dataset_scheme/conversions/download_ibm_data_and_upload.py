import argparse
import os
import random
from pathlib import Path

from huggingface_hub import HfFileSystem

from src.experiments.experiment_preparation.dataset_scheme.conversions.download_ibm_data_manager import \
    download_huggingface_files_parllel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_names', nargs='+', help='List of file_names to download', default=[])
    parser.add_argument('--input_dir', help='Output directory', default=
    "/cs/snapless/gabis/eliyahabba/ibm_results_data_full")
    parser.add_argument('--probs', type=bool, help='Whether to include probs in the schema', default=True)
    parser.add_argument('--batch_size', type=int, help='Batch size for processing parquet files', default=1000)
    parser.add_argument('--repo_name', help='Repository name for the schema files',
                        default="eliyahabba/llm-evaluation")
    parser.add_argument('--scheme_files_dir', help='Directory to store the scheme files',
                        default="/cs/snapless/gabis/eliyahabba/scheme_files")

    args = parser.parse_args()

    fs = HfFileSystem()
    existing_files = fs.ls(f"datasets/{args.repo_name}", detail=False)
    existing_files = [Path(file).stem.split("_test")[0] for file in existing_files if file.endswith('.parquet')]
    args.file_names = [file for file in os.listdir(args.input_dir) if Path(file).stem not in existing_files][::2]
    print(f"Downloading {len(args.file_names)} files")
    # random.shuffle(args.file_names)
    download_huggingface_files_parllel(input_dir=Path(args.input_dir), file_names=args.file_names,
                                       repo_name=args.repo_name,
                                       scheme_files_dir=args.scheme_files_dir, probs=args.probs)
