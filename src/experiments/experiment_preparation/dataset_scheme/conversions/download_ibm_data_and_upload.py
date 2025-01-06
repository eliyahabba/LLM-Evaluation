import argparse
from pathlib import Path

from src.experiments.experiment_preparation.dataset_scheme.conversions.download_ibm_data_manager import \
    download_huggingface_files_parllel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', nargs='+', help='List of urls to download', default=[])
    parser.add_argument('--output_dir', help='Output directory', default=
    "/cs/snapless/gabis/eliyahabba/ibm_results_data_full")
    parser.add_argument('--probs', type=bool, help='Whether to include probs in the schema', default=True)
    parser.add_argument('--batch_size', type=int, help='Batch size for processing parquet files', default=1000)
    parser.add_argument('--repo_name', help='Repository name for the schema files',
                        default="eliyahabba/llm-evaluation-ibm")
    parser.add_argument('--scheme_files_dir', help='Directory to store the scheme files',
                        default="/cs/snapless/gabis/eliyahabba/scheme_files")

    args = parser.parse_args()

    # List of URLs to download
    args.urls = [
        # "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-16T00%3A00%3A00%2B00%3A00_2024-12-17T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-17T00%3A00%3A00%2B00%3A00_2024-12-18T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-18T00%3A00%3A00%2B00%3A00_2024-12-19T00%3A00%3A00%2B00%3A00.parquet",
        # "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-19T00%3A00%3A00%2B00%3A00_2024-12-20T00%3A00%3A00%2B00%3A00.parquet",
        # "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-20T00%3A00%3A00%2B00%3A00_2024-12-21T00%3A00%3A00%2B00%3A00.parquet",
        # "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-21T00%3A00%3A00%2B00%3A00_2024-12-22T00%3A00%3A00%2B00%3A00.parquet"
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-22T08%3A00%3A00%2B00%3A00_2024-12-22T12%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-22T16%3A00%3A00%2B00%3A00_2024-12-22T18%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-22T18%3A00%3A00%2B00%3A00_2024-12-22T20%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-22T20%3A00%3A00%2B00%3A00_2024-12-22T22%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-22T22%3A00%3A00%2B00%3A00_2024-12-23T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-22T22%3A00%3A00%2B00%3A00_2024-12-23T00%3A00%3A00%2B00%3A00.parquet",
    ]
    download_huggingface_files_parllel(output_dir=Path(args.output_dir), urls=args.urls, repo_name=args.repo_name,
                                       scheme_files_dir=args.scheme_files_dir, probs=args.probs)
