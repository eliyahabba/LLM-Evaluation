from datasets import load_dataset
from huggingface_hub import HfApi

from config.get_config import Config


def delete_repository(repo_name: str, token: str):
    """
    Completely delete a Hugging Face dataset repository.

    Args:
        repo_name: Name of the repository (e.g., 'username/dataset-name')
        token: HuggingFace API token
    """
    try:
        api = HfApi()
        api.delete_repo(
            repo_id=repo_name,
            token=token,
            repo_type="dataset"
        )
        print(f"Successfully deleted repository {repo_name}")

    except Exception as e:
        print(f"Error occurred: {e}")


def remove_rows(repo_name: str, token: str):
    """
    Remove all rows from a Hugging Face dataset while preserving the repository and schema.

    Args:
        repo_name: Name of the repository (e.g., 'username/dataset-name')
        token: HuggingFace API token
    """
    try:
        # Load the dataset to get its structure
        dataset = load_dataset(repo_name, token=token)

        # Create an empty dataset with the same schema
        empty_dataset = dataset['test'].select([])

        # Upload the empty dataset back to hub
        empty_dataset.push_to_hub(
            repo_name,
            token=token,
            private=True  # Maintain the private status
        )
        print(f"Successfully removed all rows from dataset {repo_name}")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    DATA_PATH = "../converted_data.json"  # Path to your converted data
    REPO_NAME = "eliyahabba/llm-evaluation-dataset"  # Replace with your desired repo name
    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
    remove_rows(REPO_NAME, TOKEN)
    # delete_repository(REPO_NAME, TOKEN)
