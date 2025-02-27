import json
import os
import traceback
from datetime import datetime
from multiprocessing import Pool, Manager
from typing import List, Dict, Any, Optional
from huggingface_hub import login
from config.get_config import Config

config = Config()
token = config.config_values.get("hf_access_token")

login(token=token)


import pandas as pd
from datasets import load_dataset, disable_caching
from filelock import FileLock
from tqdm import tqdm
data_type = "old"
# Model configuration
MODELS = [
    'OLMoE-1B-7B-0924-Instruct',
    'Meta-Llama-3-8B-Instruct',
    'Mistral-7B-Instruct-v0.3',
    'Llama-3.2-1B-Instruct',
    'Llama-3.2-3B-Instruct',
]

# Dataset configurations
BASE_DATASETS = [
    "ai2_arc.arc_challenge",
    "ai2_arc.arc_easy",
    "hellaswag",
    "openbook_qa",
    "social_iqa",
    "race_high", "race_middle"
]

MMLU_SUBTASKS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

MMLU_PRO_SUBTASKS = [
    "history", "law", "health", "physics", "business", "other",
    "philosophy", "psychology", "economics", "math", "biology",
    "chemistry", "computer_science", "engineering",
]


def get_all_datasets() -> List[str]:
    """Compile complete list of datasets to analyze."""
    datasets = BASE_DATASETS.copy()
    datasets.extend(f"mmlu.{task}" for task in MMLU_SUBTASKS)
    datasets.extend(f"mmlu_pro.{task}" for task in MMLU_PRO_SUBTASKS)
    return datasets


# Disable caching globally
disable_caching()

# Load a specific model/language/shots benchmark
def load_benchmark(repo_id, model_name, language="en", shots=0, benchmark_file="mmlu.global_facts.parquet", data_type="old"):
    if data_type=="old":
        file_path = f"{model_name}/shots_{shots}/{language}/{benchmark_file}"
    else:
        file_path = f"{model_name}//{language}/{shots}_shot/{benchmark_file}"
    return load_dataset(repo_id, data_files=file_path, split="train", cache_dir=None,    download_mode="force_redownload")


# Examples
# Example 1: Loading from Dove_Lite repository
def get_dataset(model, shot, dataset, data_type):
    repo1 = "nlphuji/Dove_Lite"
    repo2 = "eliyahabba/llm-evaluation-analysis-split-folders"
    if data_type=="old":
        repo = repo2
        lang = "english"
        cols = ['sample_index', 'model', 'dataset', 'template', 'separator', 'enumerator', 'choices_order', 'shots']
        df = load_benchmark(repo, model, lang, shot, f"{dataset}.parquet",data_type)
        df = df.to_pandas()
        df = df.drop_duplicates(subset=cols)
        df = df[
                    (df["shots"] == 0) |
                    (~df.choices_order.isin(["correct_first", "correct_last"]))
                    ]


    elif data_type=="new":
        repo = repo1
        lang = "en"
        df = load_benchmark(repo, model, lang, shot, f"{dataset}.parquet", data_type)
    else:
        raise ValueError("Invalid data type")

    return df


def process_configuration(dataset: str, data_type) -> Optional[Dict[str, Any]]:
    """Process all models for a single dataset."""
    results = []
    output_file = f"{data_type}_dataset_sizes.csv"
    lock_file = f"{data_type}_dataset_sizes.csv.lock"

    # Create file lock
    lock = FileLock(lock_file)

    # Load existing data if file exists
    existing_data = None
    with lock:
        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file)

    total_configs = len(MODELS) * 2  # 2 shots per model
    processed = 0

    print(f"\nProcessing dataset: {dataset}")
    print(f"Total configurations for this dataset: {total_configs}")

    # Process each model
    for model in MODELS:
        for shot in [0, 5]:
            processed += 1
            print(f"Progress: {processed}/{total_configs} - Processing {model} with {shot} shots", end='\r')

            # Check if we already have this configuration
            if existing_data is not None:
                existing_row = existing_data[
                    (existing_data['model'] == model) &
                    (existing_data['dataset'] == dataset) &
                    (existing_data['shots'] == shot)
                ]
                if len(existing_row) > 0:
                    results.append({
                        'model': model,
                        'dataset': dataset,
                        'shots': shot,
                        'num_examples': existing_row.iloc[0]['num_examples']
                    })
                    continue

            # If not found in existing data, process it
            try:
                df = get_dataset(model, shot, dataset, data_type)
                num_examples = len(df)

                results.append({
                    'model': model,
                    'dataset': dataset,
                    'shots': shot,
                    'num_examples': num_examples
                })
            except Exception as e:
                print(f"\nError processing {model} - {dataset} - {shot} shots: {str(e)}")
                results.append({
                    'model': model,
                    'dataset': dataset,
                    'shots': shot,
                    'num_examples': 0,
                    'error': str(e)
                })

    print(f"\nCompleted dataset: {dataset}")

    # Convert results to DataFrame and save
    if results:  # Only save if we have new results
        results_df = pd.DataFrame(results)
        with lock:  # Use lock when writing to file
            if os.path.exists(output_file):
                results_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                results_df.to_csv(output_file, index=False)

    return None


def log_error(error: Exception, params: str) -> Dict[str, Any]:
    """Log error details for failed configurations.

    Args:
        error: The exception that occurred
        params: The dataset that failed

    Returns:
        Dict containing error details
    """
    error_message = (
        f"\n{'=' * 50}\n"
        f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Dataset that failed: {json.dumps(params, indent=2)}\n"
        f"Error message: {str(error)}\n"
        f"Traceback:\n{traceback.format_exc()}\n"
        f"{'=' * 50}\n"
    )
    print(error_message)
    return {
        'status': 'error',
        'params': params,
        'error': str(error)
    }


def process_with_error_handling(dataset: str) -> Optional[Dict[str, Any]]:
    """Wrapper for processing dataset with error handling.

    Args:
        dataset: Name of the dataset to process

    Returns:
        None if successful, error dict if failed
    """
    try:
        return process_configuration(dataset,data_type)
    except Exception as e:
        return log_error(e, dataset)


def run_configuration_analysis(num_processes: int = 1) -> None:
    """Run parallel analysis of datasets across all models."""
    datasets = get_all_datasets()
    total_datasets = len(datasets)

    print(f"Starting analysis with {num_processes} processes")
    print(f"Total datasets to process: {total_datasets}")
    print(f"Total configurations to process: {total_datasets * len(MODELS) * 2}")

    # Run parallel processing with progress bar
    with Manager() as manager:
        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_with_error_handling, datasets),
                total=total_datasets,
                desc="Processing datasets",
                unit="dataset"
            ))


def print_failed_configurations(data_type):
    """Print all configurations that failed to process."""
    output_file = f"{data_type}_dataset_sizes.csv"

    if not os.path.exists(output_file):
        print("No results file found")
        return

    df = pd.read_csv(output_file)

    # Filter for failed configurations (where num_examples is 0)
    failed_configs = df[df['num_examples'] == 0]

    if len(failed_configs) == 0:
        print("No failed configurations found")
        return

    print("\nFailed Configurations:")
    print("=" * 80)
    for _, row in failed_configs.iterrows():
        print(f"Model: {row['model']}")
        print(f"Dataset: {row['dataset']}")
        print(f"Shots: {row['shots']}")
        if 'error' in row:
            print(f"Error: {row['error']}")
        print("-" * 80)

    print(f"\nTotal failed configurations: {len(failed_configs)}")


if __name__ == "__main__":
    run_configuration_analysis(num_processes=50)
    print_failed_configurations(data_type)
