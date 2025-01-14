from datetime import datetime, timedelta
import pytz


import argparse
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path

import requests
from tqdm import tqdm

from config.get_config import Config

config = Config()
TOKEN = config.config_values.get("hf_access_token", "")


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


def download_huggingface_files_parllel(output_dir: Path, urls, repo_name, scheme_files_dir, probs=True):
    """
    Downloads parquet files from Hugging Face to a specified directory

    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_processes = min(8, cpu_count() - 1)
    logger = setup_logging()
    logger.info("Starting processing...")
    logger.info(f"Starting parallel processing with {num_processes} processes...")

    with Manager() as manager:
        process_func = partial(process_file_safe, probs=probs, output_dir=output_dir, repo_name=repo_name,
                               scheme_files_dir=scheme_files_dir, logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, urls),
                total=len(urls),
                desc="Processing files"
            ))


def process_file_safe(url, output_dir: Path, repo_name, scheme_files_dir, probs, logger):
    pid = os.getpid()
    start_time = time.time()
    try:
        logger.info(f"Process {pid} starting to process file {url}")
        procces_file(url, output_dir=output_dir, repo_name=repo_name, scheme_files_dir=scheme_files_dir, probs=probs,
                     logger=logger)
    except Exception as e:
        logger.error(f"Process {pid} encountered error processing file {url}: {e}")
        return None
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Process {pid} finished processing {url}. Processing time: {elapsed_time:.2f} seconds")


def procces_file(url, output_dir: Path, repo_name, scheme_files_dir, probs, logger):
    # Extract date from URL for the output filename
    # Example: from URL get '2024-12-16' for the filename
    original_filename = url.split('/')[-1]
    output_path = output_dir / original_filename
    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
    else:
        try:
            logger.info(f"Downloading {original_filename}...")

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name

                response = requests.get(url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            shutil.move(temp_path, output_path)

            logger.info(f"Download completed: {original_filename}")
            print(f"Download completed: {original_filename}")

        except (requests.exceptions.RequestException, IOError) as e:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            print(f"Error downloading {original_filename}: {e}")
            logger.error(f"Error downloading {original_filename}: {e}")


def generate_file_names(start_time, end_time):
    file_names = []

    current_time = start_time
    while current_time <= end_time:
        file_name = f"data_{current_time.isoformat()}.parquet"
        file_names.append(file_name)
        current_time += timedelta(hours=1)

    return file_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', nargs='+', help='List of urls to download')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--probs', type=bool, help='Whether to include probs in the schema')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing parquet files', default=1000)
    parser.add_argument('--repo_name', help='Repository name for the schema files')
    parser.add_argument('--scheme_files_dir', help='Directory to store the scheme files')
    args = parser.parse_args()
    # Set start date
    # REPO_NAME = "eliyahabba/llm-evaluation-without-probs"  # Replace with your desired repo name
    # scheme_files_dir = "/cs/snapless/gabis/eliyahabba/scheme_files_without_probs"
    # output_directory = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full"
    # parquet_path = Path("~/Downloads/data_sample.parquet")
    # convert_to_scheme_format(parquet_path, batch_size=100)
    main_path = 'https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/'
    # List of URLs to download
    start_time = datetime(2025, 1, 11, 19, 0, tzinfo=pytz.UTC)

    end_time = datetime(2025, 1, 12, 19, 0, tzinfo=pytz.UTC)

    files = generate_file_names(start_time, end_time)
    urls = [f"{main_path}{file}" for file in files]
    for file in urls:
        print(file)
