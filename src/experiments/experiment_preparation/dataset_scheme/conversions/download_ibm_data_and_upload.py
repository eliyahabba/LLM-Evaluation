import os
import tempfile
from datetime import datetime
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


def convert_to_scheme_format(parquet_path, batch_size=1000):
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
    schema = pa.Table.from_pylist(first_converted).schema

    # Create the ParquetWriter
    writer = pq.ParquetWriter(temp_path, schema)

    # Write the first batch
    table = pa.Table.from_pylist(first_converted)
    writer.write_table(table)

    # Process remaining batches
    for i, batch in tqdm(enumerate(processor.process_batches()), desc="Processing batches"):
        print(f"Processing batch {i + 1}...")  # i+1 because we already processed first batch
        converted_data = converter.convert_dataframe(batch)

        # Convert to table and write
        table = pa.Table.from_pylist(converted_data)
        writer.write_table(table)

    # Close the writer
    writer.close()

    print(f"Data saved to: {temp_path}")
    REPO_NAME = "eliyahabba/llm-evaluation-dataset-ibm"  # Replace with your desired repo name
    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
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
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-16T00%3A00%3A00%2B00%3A00_2024-12-17T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-17T00%3A00%3A00%2B00%3A00_2024-12-18T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-18T00%3A00%3A00%2B00%3A00_2024-12-19T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-19T00%3A00%3A00%2B00%3A00_2024-12-20T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-20T00%3A00%3A00%2B00%3A00_2024-12-21T00%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2024-12-21T00%3A00%3A00%2B00%3A00_2024-12-22T00%3A00%3A00%2B00%3A00.parquet"
     ]

    for url in urls:
        # Extract date from URL for the output filename
        # Example: from URL get '2024-12-16' for the filename
        original_filename = url.split('/')[-1]
        output_path = Path(os.path.join(output_dir, original_filename))

        try:
            print(f"Downloading {original_filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Download completed: {original_filename}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {original_filename}: {e}")

        convert_to_scheme_format(output_path)
        # now load the file by chu

if __name__ == "__main__":
    # Set start date
    output_directory = "/cs/snapless/gabis/eliyahabba/ibm_results_data"

    download_huggingface_files(output_directory)
    # parquet_path = Path("~/Downloads/data_sample.parquet")
    # convert_to_scheme_format(parquet_path, batch_size=1000)
