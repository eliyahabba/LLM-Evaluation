import json
import logging
import os
import tempfile
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

from config.get_config import Config
from src.experiments.experiment_preparation.dataset_scheme.conversions.RunOutputMerger import RunOutputMerger
from src.experiments.experiment_preparation.dataset_scheme.conversions.dataset_schema_adapter_ibm import SchemaConverter
from src.utils.Constants import Constants

REPO_NAME = "eliyahabba/llm-evaluation-ibm"  # Replace with your desired repo name
config = Config()
TOKEN = config.config_values.get("hf_access_token", "")


def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'processing_log_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_model_info_schema():
    return pa.struct([
        ('name', pa.string()),
        ('family', pa.string())  # nullable by default in PyArrow
    ])


def create_model_config_schema():
    return pa.struct([
        ('architecture', pa.string()),
        ('parameters', pa.int64()),
        ('context_window', pa.int64()),
        ('is_instruct', pa.bool_()),
        ('hf_path', pa.string()),
        ('revision', pa.string())
    ])


def create_inference_settings_schema():
    quantization_schema = pa.struct([
        ('bit_precision', pa.string()),
        ('method', pa.string())
    ])

    generation_args_schema = pa.struct([
        ('use_vllm', pa.bool_()),
        ('temperature', pa.null()),  # או float64() אם צריך
        ('top_p', pa.null()),
        ('top_k', pa.int64()),
        ('max_tokens', pa.int64()),
        ('stop_sequences', pa.list_(pa.string()))
    ])

    return pa.struct([
        ('quantization', quantization_schema),
        ('generation_args', generation_args_schema)
    ])


def create_choice_schema():
    return pa.struct([
        ('id', pa.string()),
        ('text', pa.string())
    ])


def create_demos_schema():
    return pa.list_(
        pa.struct([
            ('topic', pa.string()),
            ('question', pa.string()),
            ('choices', pa.list_(create_choice_schema())),
            ('answer', pa.int64())
        ]),
    )


def create_prompt_config_schema():
    choices_order_schema = pa.struct([
        ('method', pa.string()),
        ('description', pa.string())
    ])

    format_schema = pa.struct([
        ('type', pa.string()),
        ('template', pa.string()),
        ('separator', pa.string()),
        ('enumerator', pa.string()),
        ('choices_order', choices_order_schema),
        ('shots', pa.int64()),
        ('demos', create_demos_schema())
    ])

    return pa.struct([
        ('prompt_class', pa.string()),
        ('format', format_schema)
    ])


def create_token_info_schema():
    return pa.struct([
        ('token_index', pa.int64()),
        ('logprob', pa.float64()),
        ('rank', pa.int64()),
        ('decoded_token', pa.string())
    ])


def create_instance_schema():
    classification_fields_schema = pa.struct([
        ('question', pa.string()),
        ('choices', pa.list_(create_choice_schema())),
        ('ground_truth', create_choice_schema())
    ])

    sample_identifier_schema = pa.struct([
        ('dataset_name', pa.string()),
        ('split', pa.string()),
        ('hf_repo', pa.string()),
        ('hf_index', pa.int64())
    ])

    return pa.struct([
        ('task_type', pa.string()),
        ('raw_input', pa.list_(pa.struct([
            ('role', pa.string()),
            ('content', pa.string())
        ]))),
        ('sample_identifier', sample_identifier_schema),
        ('perplexity', pa.float64()),
        ('classification_fields', classification_fields_schema),
        ('prompt_logprobs', pa.list_(pa.list_(create_token_info_schema())))
    ])


def create_output_schema():
    return pa.struct([
        ('response', pa.string()),
        ('cumulative_logprob', pa.float64()),
        ('generated_tokens_ids', pa.list_(pa.int64())),
        ('generated_tokens_logprobs', pa.list_(pa.list_(create_token_info_schema()))),
        ('max_prob', pa.string())
    ])


def create_evaluation_schema():
    evaluation_method_schema = pa.struct([
        ('method_name', pa.string()),
        ('description', pa.string())
    ])

    return pa.struct([
        ('evaluation_method', evaluation_method_schema),
        ('ground_truth', pa.string()),
        ('score', pa.float64())
    ])


def create_full_schema():
    return pa.schema([
        ('evaluation_id', pa.string()),
        ('model', pa.struct([
            ('model_info', create_model_info_schema()),
            ('configuration', create_model_config_schema()),
            ('inference_settings', create_inference_settings_schema())
        ])),
        ('prompt_config', create_prompt_config_schema()),
        ('instance', create_instance_schema()),
        ('output', create_output_schema()),
        ('evaluation', create_evaluation_schema())
    ])


def convert_to_scheme_format(parquet_path, batch_size=1000, logger=None):
    models_metadata_path = Path(Constants.ExperimentConstants.MODELS_METADATA_PATH)
    converter = SchemaConverter(models_metadata_path)
    processor = RunOutputMerger(parquet_path, batch_size)

    # Dictionary to store data for different splits
    split_data = {}
    temp_files = {}
    writers = {}
    schema = create_full_schema()

    def get_writer_for_split(split):
        if split not in writers:
            # Create a new temporary file for this split
            temp_path = tempfile.mktemp(suffix=f'_{split}.parquet')
            temp_files[split] = temp_path
            writers[split] = pq.ParquetWriter(temp_path, schema)
        return writers[split]

    def write_rows(rows, split, batch_num=None):
        if not rows:
            return

        writer = get_writer_for_split(split)
        try:
            table = pa.Table.from_pylist(rows)
            writer.write_table(table)
            if logger:
                logger.info(f"Batch {batch_num} written successfully for split {split}")
        except Exception as e:
            if logger:
                logger.error(f"Error writing batch for split {split}: {e}")
            # Try writing row by row to identify problematic rows
            for i, row in enumerate(rows):
                try:
                    table = pa.Table.from_pylist([row], schema)
                    writer.write_table(table)
                except Exception as e:
                    if logger:
                        logger.error(f"Error writing row {i} in split {split}: {e}")
                    print(f"Error writing row {i} in split {split}: {e}")
                    print(json.dumps(row, indent=4))

    # Process first batch
    first_batch = next(processor.process_batches())
    first_converted = converter.convert_dataframe(first_batch,logger=logger)

    # Separate first batch by splits
    first_batch_splits = {}
    for item in first_converted:
        split = item['instance']['sample_identifier']['split']
        if split not in first_batch_splits:
            first_batch_splits[split] = []
        first_batch_splits[split].append(item)

    # Write first batch for each split
    for split, items in first_batch_splits.items():
        write_rows(items, split, batch_num=0)

    # Process remaining batches
    for i, batch in tqdm(enumerate(processor.process_batches()), desc="Processing batches"):
        if logger:
            logger.info(f"Processing batch {i + 1}...")
        print(f"Processing batch {i + 1}...")

        converted_data = converter.convert_dataframe(batch,logger=logger)

        # Separate batch by splits
        batch_splits = {}
        for item in converted_data:
            split = item['instance']['sample_identifier']['split']
            if split not in batch_splits:
                batch_splits[split] = []
            batch_splits[split].append(item)

        # Write each split's data
        for split, items in batch_splits.items():
            write_rows(items, split, batch_num=i + 1)

    # Close all writers
    for writer in writers.values():
        writer.close()


    try:
        create_repo(
            repo_id=REPO_NAME,
            token=TOKEN,
            private=False,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    api = HfApi()
    base_filename = parquet_path.stem  # Get filename without extension

    # Upload each split file
    for split, temp_path in temp_files.items():
        filename = f"{base_filename}_{split}.parquet"
        try:
            print(f"Uploading {split} file to Hugging Face...")
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=REPO_NAME,
                token=TOKEN,
                repo_type="dataset"
            )
            print(f"Successfully uploaded {split} file as {filename} to {REPO_NAME}")
        except Exception as e:
            print(f"Error uploading {split} file: {e}")
        finally:
            # Clean up temporary file
            if Path(temp_path).exists():
                Path(temp_path).unlink()
                print(f"Temporary file for {split} cleaned up")

def convert_to_scheme_format1(parquet_path,batch_size=1000, logger=None):
    models_metadata_path = Path(Constants.ExperimentConstants.MODELS_METADATA_PATH)

    # Initialize converter
    converter = SchemaConverter(models_metadata_path)

    # Process data in batches
    processor = RunOutputMerger(parquet_path, batch_size)

    temp_path = tempfile.mktemp(suffix='.parquet')

    # Initialize the writer with the schema of your data
    # We'll get this from the first batch
    first_batch = next(processor.process_batches())
    first_converted = converter.convert_dataframe(first_batch)

    schema = create_full_schema()

    # Create the ParquetWriter
    writer = pq.ParquetWriter(temp_path, schema)

    # Write the first batch
    table = pa.Table.from_pylist(first_converted)
    try:
        writer.write_table(table)
        if logger:
            logger.info(f"First batch written successfully")
    except Exception as e:
        if logger:
            logger.error(f"Error writing first batch: {e}")
        # try on each row in the first batch to find the problematic row
        for i, row in enumerate(first_converted):
            try:
                table = pa.Table.from_pylist([row], schema)
                writer.write_table(table)
            except Exception as e:
                if logger:
                    logger.error(f"Error writing row {i}: {e}")
                print(f"Error writing row {i}: {e}")
                # print the row in prrety format
                print(json.dumps(row, indent=4))

    # Process remaining batches
    for i, batch in tqdm(enumerate(processor.process_batches()), desc="Processing batches"):
        if logger:
            logger.info(f"Processing batch {i + 1}...")
        print(f"Processing batch {i + 1}...")  # i+1 because we already processed first batch
        converted_data = converter.convert_dataframe(batch)

        # Convert to table and write
        table = pa.Table.from_pylist(converted_data)
        try:
            writer.write_table(table)
        except Exception as e:
            if logger:
                logger.error(f"Error writing batch {i + 1}: {e}")
            # try on each row in the first batch to find the problematic row
            for i, row in enumerate(converted_data):
                try:
                    table = pa.Table.from_pylist([row], schema)
                    writer.write_table(table)
                except Exception as e:
                    if logger:
                        logger.error(f"Error writing row {i}: {e}")
                    print(f"Error writing row {i}: {e}")
                    # print the row in prrety format
                    print(json.dumps(row, indent=4))

    # Close the writer
    if logger:
        logger.info(f"Closing writer... for {parquet_path}")
    writer.close()

    print(f"Data saved to: {temp_path}")
    try:
        create_repo(
            repo_id=REPO_NAME,
            token=TOKEN,
            private=False,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    # Upload to Hugging Face
    api = HfApi()
    filename = parquet_path.name

    # Upload to Hugging Face
    try:
        print(f"Uploading file to Hugging Face...")
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=filename,
            repo_id=REPO_NAME,
            token=TOKEN,
            repo_type="dataset"
        )
        print(f"Successfully uploaded file as {filename} to {REPO_NAME}")
    except Exception as e:
        print(f"Error uploading file: {e}")
    finally:
        # Clean up temporary file
        if Path(temp_path).exists():
            Path(temp_path).unlink()
            print("Temporary file cleaned up")


def download_huggingface_files(output_dir):
    """
    Downloads parquet files from Hugging Face to a specified directory

    Args:
        start_date (datetime): Start date
        num_days (int): Number of days to download
        output_dir (str): Path to output directory
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List of URLs to download
    urls = [
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-16.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-16T00%3A00%3A00%2B00%3A00_2024-12-17T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-17T00%3A00%3A00%2B00%3A00_2024-12-18T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-18T00%3A00%3A00%2B00%3A00_2024-12-19T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-19T00%3A00%3A00%2B00%3A00_2024-12-20T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-20T00%3A00%3A00%2B00%3A00_2024-12-21T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-21T00%3A00%3A00%2B00%3A00_2024-12-22T00%3A00%3A00%2B00%3A00.parquet"
    ]

    for url in urls:
        procces_file(url, output_dir)


def download_huggingface_files_parllel(output_dir):
    """
    Downloads parquet files from Hugging Face to a specified directory

    Args:
        start_date (datetime): Start date
        num_days (int): Number of days to download
        output_dir (str): Path to output directory
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List of URLs to download
    urls = [
        #"https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-16T00%3A00%3A00%2B00%3A00_2024-12-17T00%3A00%3A00%2B00%3A00.parquet",
        #"https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-17T00%3A00%3A00%2B00%3A00_2024-12-18T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-18T00%3A00%3A00%2B00%3A00_2024-12-19T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-19T00%3A00%3A00%2B00%3A00_2024-12-20T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-20T00%3A00%3A00%2B00%3A00_2024-12-21T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-21T00%3A00%3A00%2B00%3A00_2024-12-22T00%3A00%3A00%2B00%3A00.parquet"
    ]

    num_processes = max(1, cpu_count() - 1)
    logger = setup_logging()
    logger.info("Starting processing...")
    logger.info(f"Starting parallel processing with {num_processes} processes...")

    with Manager() as manager:
        process_func = partial(procces_file,
                               output_dir=output_dir,
                               logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, urls),
                total=len(urls),
                desc="Processing files"
            ))


def procces_file(url, output_dir,logger):
    # Extract date from URL for the output filename
    # Example: from URL get '2024-12-16' for the filename
    original_filename = url.split('/')[-1]
    output_path = Path(os.path.join(output_dir, original_filename))
    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        print(f"File already exists: {output_path}")
    else:
        try:
            logger.info(f"Downloading {original_filename}...")
            print(f"Downloading {original_filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Download completed: {original_filename}")
            print(f"Download completed: {original_filename}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {original_filename}: {e}")
            logger.error(f"Error downloading {original_filename}: {e}")


if __name__ == "__main__":
    # Set start date
    output_directory = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full"
    download_huggingface_files_parllel(output_directory)
