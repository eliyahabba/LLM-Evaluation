from huggingface_hub import HfApi

from config.get_config import Config


def update_readme(repo_name: str, token: str):
    """
    Update the README.md of a HuggingFace dataset repository.

    Args:
        repo_name: Name of the repository (e.g., 'username/dataset-name')
        token: HuggingFace API token
    """
    readme_content = """
# LLM Evaluation Dataset

This dataset contains evaluation results of Large Language Models on multiple choice questions.

## Dataset Structure

The dataset contains the following main components:
- Model configuration (name, quantization settings)
- Prompt information (template, formatting, token IDs)
- Instance details (questions, choices, metadata)
- Model outputs (responses, probabilities, token-level information)
- Evaluation metrics (ground truth, scores)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("eliyahabba/llm-evaluation-dataset")

# Access the data
test_data = dataset['test']
```

## Features

- Comprehensive evaluation metrics
- Token-level probability information
- Detailed model responses
- Rich metadata for each instance

## Schema Details

For detailed schema information, please refer to our Github repository: [link]

## Citation

If you use this dataset, please cite:
[TODO: Add citation]
    """

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )
        print(f"Successfully updated README for {repo_name}")

    except Exception as e:
        print(f"Error updating README: {e}")


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "../converted_data.json"  # Path to your converted data
    REPO_NAME = "eliyahabba/llm-evaluation-dataset"  # Replace with your desired repo name
    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")

    update_readme(REPO_NAME, TOKEN)
