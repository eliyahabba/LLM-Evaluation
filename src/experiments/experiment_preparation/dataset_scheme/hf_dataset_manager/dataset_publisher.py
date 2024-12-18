import json
from collections import defaultdict
from pathlib import Path

from datasets import DatasetDict, Dataset, Features, Sequence, Value
from huggingface_hub import HfApi, create_repo

from config.get_config import Config


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
            'Instruct Template': Value('string'),
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
            'CumulativeLogprob': Value('float32'),
            'GeneratedTokenIds': Sequence(Value('int64')),
            'TokenLogprobs_tokens': Sequence(Value('string')),
            'TokenLogprobs_ids': Sequence(Value('int64')),
            'top5_tokens': Sequence(Value('string')),
            'top5_token_ids': Sequence(Value('int64')),
            'top5_logprobs': Sequence(Value('float32')),
            'top5_ranks': Sequence(Value('int64')),
            'MaxProb': Value('string')
        },
        'Evaluation': {
            'GroundTruth': Value('string'),
            'Score': Value('float32')
        }
    })


def read_data(
        data_path: str,
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
    upload_to_huggingface(data, repo_name, token, private)
def upload_to_huggingface(
        data: list,
        repo_name: str,
        token: str,
        private: bool = False
) -> None:
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")

    # Convert to dataset
    split_data = defaultdict(list)
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
        upload_to_huggingface(
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
