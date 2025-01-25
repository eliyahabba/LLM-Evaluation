import argparse
import logging
import os
import random
import shutil
import time
from datetime import datetime
from datetime import timedelta
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path

import requests
from huggingface_hub import HfFileSystem
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
    logger.info(f"Downloading {len(urls)} files to {output_dir}")

    with Manager() as manager:
        process_func = partial(process_file_safe, probs=probs, output_dir=output_dir, repo_name=repo_name,
                               scheme_files_dir=scheme_files_dir, logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, urls),
                total=len(urls),
                desc="Processing files"
            ))
    # remove temp directory if it exists
    temp_dir = output_dir / 'temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


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

            temp_dir = output_path.parent / 'temp'
            os.makedirs(temp_dir, exist_ok=True)
            temp_filename = f'temp_{original_filename}'
            temp_path = os.path.join(temp_dir, temp_filename)

            with open(temp_path, 'wb') as temp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            shutil.move(temp_path, output_path)
            logger.info(f"Downloaded {original_filename} successfully")

        except Exception as e:
            logger.error(f"Error downloading {original_filename}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


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

    args.old_output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full"
    args.output_dir = "/cs/labs/gabis/eliyahabba/ibm_results_data_full"
    os.makedirs(args.output_dir, exist_ok=True)


    # parquet_path = Path("~/Downloads/data_sample.parquet")
    main_path = 'https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/'

    # List of URLs to download
    fs = HfFileSystem()

    existing_files = fs.ls(f"datasets/OfirArviv/HujiCollabOutput", detail=False)
    existing_files = [file.split("/")[-1] for file in existing_files if file.endswith('.parquet')]
    existing_files = [file for file in existing_files if not os.path.exists(f"{args.output_dir}/{file}")
                      and not os.path.exists(f"{args.old_output_dir}/{file}")
                      ]
    # filter the file that not already exist in the output directory
    args.urls = [f"{main_path}{file}" for file in existing_files]

    random.shuffle(args.urls)
    download_huggingface_files_parllel(output_dir=Path(args.output_dir), urls=args.urls, repo_name=args.repo_name,
                                       scheme_files_dir=args.scheme_files_dir, probs=args.probs)
