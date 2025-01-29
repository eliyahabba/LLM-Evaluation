import argparse
import logging
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, create_repo, HfFileSystem
from tqdm import tqdm

from config.get_config import Config

config = Config()
TOKEN = config.config_values.get("hf_access_token", "")
repo_name = "eliyahabba/llm-evaluation-analysis-split"
config = Config()
TOKEN = config.config_values.get("hf_access_token", "")


def get_parquet_files_from_hf(repo_id: str, token: str = None) -> List[Tuple[str, str]]:
    """
    Lists all parquet files in a Hugging Face repository

    Args:
        repo_id: Repository ID (e.g., 'username/repo-name')
        token: HF API token for private repos (optional)

    Returns:
        list: List of tuples (download_url, filename)
    """
    api = HfApi(token=token)
    try:
        # List all files in the repository
        fs = HfFileSystem()
        existing_files = fs.ls(f"datasets/{repo_id}", detail=False)
        main_path = 'https://huggingface.co/datasets'
        # Filter parquet files and create download URLs
        parquet_files = []
        for file in existing_files:
            if file.endswith('.parquet'):
                name = file.split("/")[-1]
                url = f"{main_path}/{repo_id}/resolve/main/{name}"
                parquet_files.append((url, file))

        return parquet_files
    except Exception as e:
        print(f"Error listing repository files: {str(e)}")
        return []

def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'processing_log_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def process_file(process_file_path, logger):
    """Main function to demonstrate usage."""
    logger.info(f"Processing file: {process_file_path}")
    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=process_file_path,
            path_in_repo=Path(process_file_path).name,
            repo_id=repo_name,
            token=TOKEN,
            repo_type="dataset"
        )
        logger.info(f"Successfully uploaded {Path(process_file_path).name}")
    except Exception as e:
        logger.info(f"Error uploading {Path(process_file_path).name}: {e}")


def download_huggingface_files_parllel(process_output_dir):
    """
    Downloads parquet files from Hugging Face to a specified directory

    """
    num_processes = min(24, cpu_count() - 1)
    logger = setup_logging()
    logger.info("Starting processing...")
    logger.info(f"Starting parallel processing with {num_processes} processes...")
    files = os.listdir(process_output_dir)
    files = [os.path.join(process_output_dir, file) for file in files]
    print(f"Processing {len(files)} files")
    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
    ex_files = get_parquet_files_from_hf(repo_name, TOKEN)
    ex_files = [file[1].split("/")[-1] for file in ex_files]
    files = [file for file in files if Path(file).stem not in ex_files]
    with Manager() as manager:
        process_func = partial(process_file_safe, logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, files),
                total=len(files),
                desc="Processing files"
            ))


def process_file_safe(file, logger):
    pid = os.getpid()
    start_time = time.time()
    try:
        logger.info(f"Process {pid} starting to process file {file}")
        process_file(file,
                     logger=logger)
    except Exception as e:
        logger.error(f"Process {pid} encountered error processing file {file}: {e}")
        return None
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Process {pid} finished processing {file}. Processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    process_output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split"
    # parquet_path = Path("~/Downloads/data_sample.parquet")
    # convert_to_scheme_format(parquet_path, batch_size=100)
    os.makedirs(process_output_dir, exist_ok=True)
    try:
        create_repo(
            repo_id=repo_name,
            token=TOKEN,
            private=True,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    download_huggingface_files_parllel(process_output_dir=process_output_dir)
