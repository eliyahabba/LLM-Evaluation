import argparse
import logging
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

from config.get_config import Config

config = Config()
TOKEN = config.config_values.get("hf_access_token", "")
repo_name = "eliyahabba/llm-evaluation-analysis"


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
    process_output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed"
    # parquet_path = Path("~/Downloads/data_sample.parquet")
    # convert_to_scheme_format(parquet_path, batch_size=100)
    os.makedirs(process_output_dir, exist_ok=True)
    download_huggingface_files_parllel(process_output_dir=process_output_dir)
