import tempfile
import time
from pathlib import Path

from datasets import Features, Sequence, Value

from config.get_config import Config
from typing import List, Dict
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
from collections import defaultdict
import tempfile
import time
import pyarrow as pa
import pyarrow.parquet as pq

def define_features():
    """Define the features schema for the dataset."""
    return Features({
        'Model': {
            'Name': Value('string'),
            'Quantization': {
                'QuantizationBit': Value('string'),
                'QuantizationMethod': Value('string')
            }
        },
        'Prompt': {
            'Template': Value('string'),
            'Separator': Value('string'),
            'Enumerator': Value('string'),
            'ShuffleChoices': Value('bool'),
            'Shots': Value('string'),
            'TokenIds': Sequence(Value('int64'))
        },
        'Instance': {
            'RawInput': Value('string'),
            'SampleIdentifier': {
                'dataset_name': Value('string'),
                'split': Value('string'),
                'hf_repo': Value('string'),
                'hf_index': Value('int64')
            },
            'Topic': Value('string'),
            'Question': Value('string'),
            'Choices': Value('string'),
            'Perplexity': Value('float32'),
            'GroundTruth': Value('string')
        },
        'Output': {
            'Response': Value('string'),
            'CumulativeLogProb': Value('float32'),
            'GeneratedTokensIds': Sequence(Value('int64')),
            'GeneratedTokensLogProbs_tokens': Sequence(Value('string')),
            'GeneratedTokensLogProbs_ids': Sequence(Value('int64')),
            'top5_tokens': Sequence(Value('string')),
            'top5_token_ids': Sequence(Value('int64')),
            'top5_LogProbs': Sequence(Value('float32')),
            'top5_ranks': Sequence(Value('int64')),
            'MaxProb': Value('string')
        },
        'Evaluation': {
            'GroundTruth': Value('string'),
            'Score': Value('float32')
        }
    })


def read_data(
        data_path: Path,
        repo_name: str,
        token: str,
        private: bool = False
) -> None:
    """
    Upload the dataset to Hugging Face.

    Args:
        data_path: Path to the JSON file containing converted data
        repo_name: Name for the HuggingFace repository (e.g., 'username/dataset-name')
        token: HuggingFace API token
        private: Whether to create a private repository
    """
    # Load the converted data
    with open(data_path, 'r') as f:
        data = json.load(f)
    check_and_upload_to_huggingface(data, repo_name, token, private)


import json
from collections import defaultdict
from pathlib import Path
from typing import Set, List, Tuple
from datasets import DatasetDict, Dataset, load_dataset
from huggingface_hub import HfApi, create_repo


def load_existing_dataset(repo_name: str) -> Tuple[Set[str], int]:
    """
    Load existing dataset and get evaluation IDs.

    Args:
        repo_name: Name of the HuggingFace repository

    Returns:
        Tuple containing set of existing IDs and total example count
    """
    try:
        existing_dataset = load_dataset(repo_name)
        total_examples = sum(len(split) for split in existing_dataset.values())
        print(f"Found existing dataset with {total_examples} examples")

        existing_ids = set()
        for split in existing_dataset.values():
            for item in split:
                if 'EvaluationId' in item:
                    existing_ids.add(item['EvaluationId'])

        return existing_ids, total_examples

    except Exception as e:
        print(f"No existing dataset found or error occurred: {e}")
        print("Will create new dataset")
        return set(), 0


def filter_duplicates(data: List[dict], existing_ids: Set[str]) -> List[dict]:
    """
    Filter out duplicate examples based on EvaluationId.

    Args:
        data: List of data items
        existing_ids: Set of existing evaluation IDs

    Returns:
        Filtered list of data items
    """
    filtered_data = []
    for item in data:
        if item['EvaluationId'] not in existing_ids:
            filtered_data.append(item)

    duplicates = len(data) - len(filtered_data)
    print(f"Found {duplicates} duplicate examples")
    return filtered_data


def check_and_upload_parq_file_to_huggingface(
        data: List[dict],
        repo_name: str,
        token: str,
        private: bool = False,
        file_name: str = None
) -> None:
    """
    Main function to check for duplicates and upload dataset as Parquet.

    Args:
        data: List of data items to upload
        repo_name: Name for the HuggingFace repository
        token: HuggingFace API token
        private: Whether to create a private repository
        file_name: Base name for the uploaded file (optional)
    """
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    # Load existing dataset and get IDs
    # existing_ids, _ = load_existing_dataset(repo_name)

    # Filter duplicates
    # filtered_data = filter_duplicates(data, existing_ids)
    filtered_data = data
    if not filtered_data:
        print("No new data to upload")
        return

    # Split data by splits (train/test/validation)
    split_data = defaultdict(list)
    for item in filtered_data:
        split = item['instance']['sample_identifier']['split']
        split_data[split].append(item)

    # Create datasets for each split
    datasets = {
        split: Dataset.from_list(items)
        for split, items in split_data.items()
    }

    dataset_dict = DatasetDict(datasets)
    api = HfApi()

    # Create temporary Parquet file and upload
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        for split_name, dataset in dataset_dict.items():
            # Convert to Arrow table
            arrow_table = pa.Table.from_pylist(dataset.to_list())

            # Write to Parquet
            pq.write_table(arrow_table, tmp.name)

            # Generate filename with timestamp
            timestamp = int(time.time())
            file_name = file_name or "data"
            remote_path = f"{file_name}_{split_name}_{timestamp}.parquet"

            # Upload to Hugging Face
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=remote_path,
                repo_id=repo_name,
                repo_type="dataset",
                token=token
            )
            print(f"Uploaded {remote_path} to {repo_name}")

def check_and_upload_file_to_huggingface(
        data: List[dict],
        repo_name: str,
        token: str,
        private: bool = False,
        file_name: str = None
) -> None:
    """
    Main function to check for duplicates and upload dataset.

    Args:
        data: List of data items to upload
        repo_name: Name for the HuggingFace repository
        token: HuggingFace API token
        private: Whether to create a private repository
    """
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    # Load existing dataset and get IDs
    existing_ids, _ = load_existing_dataset(repo_name)

    # Filter duplicates
    filtered_data = filter_duplicates(data, existing_ids)

    if not filtered_data:
        print("No new data to upload")
        return
    # Create repository if it doesn't exist
    # Convert to dataset
    split_data = defaultdict(list)
    data = filtered_data
    for item in data:
        split = item['Instance']['SampleIdentifier']['split']
        split_data[split].append(item)

    # Create datasets for each split
    datasets = {
        split: Dataset.from_list(items)
        for split, items in split_data.items()
    }

    dataset_dict = DatasetDict(datasets)
    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_name,
            token=token,
            repo_type="dataset",
            private=private
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        for split_name, dataset in dataset_dict.items():
            dataset.to_json(tmp.name)

        timestamp = int(time.time())
        file_name = file_name or f"data"
        api.upload_file(
            path_or_fileobj=tmp.name,
            path_in_repo=f"{file_name}_{timestamp}.json",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )


def check_and_upload_to_huggingface(
        data: List[dict],
        repo_name: str,
        token: str,
        private: bool = False
) -> None:
    """
    Main function to check for duplicates and upload dataset.

    Args:
        data: List of data items to upload
        repo_name: Name for the HuggingFace repository
        token: HuggingFace API token
        private: Whether to create a private repository
    """
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    # Load existing dataset and get IDs
    existing_ids, _ = load_existing_dataset(repo_name)

    # Filter duplicates
    filtered_data = filter_duplicates(data, existing_ids)

    if not filtered_data:
        print("No new data to upload")
        return
    # Create repository if it doesn't exist
    # Convert to dataset
    split_data = defaultdict(list)
    data = filtered_data
    for item in data:
        split = item['Instance']['SampleIdentifier']['split']
        split_data[split].append(item)

    # Create datasets for each split
    datasets = {
        split: Dataset.from_list(items)
        for split, items in split_data.items()
    }

    dataset_dict = DatasetDict(datasets)

    # Push to hub
    dataset_dict.push_to_hub(
        repo_name,
        token=token,
        private=private
    )


def main():
    # Configuration
    DATA_PATH = Path(__file__).parents[1] / "converted_data.json"  # Path to your converted data
    REPO_NAME = "eliyahabba/llm-evaluation-dataset"  # Replace with your desired repo name
    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")

    # Create README content
    readme_content = """
# LLM Evaluation Dataset

This dataset contains evaluation results for Large Language Models on multiple choice questions.

## Dataset Structure

The dataset contains the following main components:
- Model configuration
- Prompt information
- Instance details
- Model outputs
- Evaluation metrics

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("eliyahabba/llm-evaluation-dataset")
```

## Citation

If you use this dataset, please cite:
[TODO: Add citation]
    """

    try:
        # Upload dataset
        read_data(
            data_path=DATA_PATH,
            repo_name=REPO_NAME,
            token=TOKEN,
            private=False  # Set to True if you want a private repository
        )

        # Update README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_NAME,
            repo_type="dataset",
            token=TOKEN
        )

        print(f"Successfully uploaded dataset to {REPO_NAME}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
