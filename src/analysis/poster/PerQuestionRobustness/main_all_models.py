import json
import os
import traceback
from datetime import datetime
from multiprocessing import Pool, Manager
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from src.analysis.create_plots.get_example_from_row import InstanceLoader
from src.analysis.paper.PerQuestionRobustness.DataLoader import DataLoader
from src.analysis.paper.PerQuestionRobustness.PromptQuestionAnalyzer import PromptQuestionAnalyzer

# Model configuration
MODELS = [
    'allenai/OLMoE-1B-7B-0924-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]

# Dataset configurations
BASE_DATASETS = [
    "ai2_arc.arc_challenge",
    "ai2_arc.arc_easy",
    "hellaswag",
    # "openbook_qa",
    # "social_iqa",
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


def process_configuration(dataset: str) -> Optional[Dict[str, Any]]:
    """Process all models for a single dataset.

    Args:
        dataset: Name of the dataset to process

    Returns:
        None if successful, error dict if failed
    """
    dfs = []
    prompt_analyzer = PromptQuestionAnalyzer()
    data_loader = DataLoader()

    # Process each model
    for model in MODELS:
        df = data_loader.load_and_process_data(
            model_name=model,
            datasets=[dataset],
            drop=False
        )

        df = df[
            (df["shots"] == 0) |
            (~df.choices_order.isin(["correct_first", "correct_last"]))
            ]
        filter_df_with_closest_answer_ind  = InstanceLoader.get_example_from_index(dataset, df)
        if df.empty:
            return None
        dfs.append(df)

    # Combine all model data
    # combined_df = pd.concat(dfs)
    #
    # # Process results
    # base_results_dir = os.path.abspath("../app/results_local")
    # os.makedirs(base_results_dir, exist_ok=True)
    #
    # prompt_analyzer.process_and_visualize_questions_all_models(
    #     df=combined_df,
    #     dataset=dataset,
    #     base_results_dir=base_results_dir
    # )
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
        return process_configuration(dataset)
    except Exception as e:
        return log_error(e, dataset)


def run_configuration_analysis(num_processes: int = 1) -> None:
    """Run parallel analysis of datasets across all models.

    Args:
        num_processes: Number of parallel processes to use
    """
    datasets = get_all_datasets()

    # Run parallel processing with progress bar
    with Manager() as manager:
        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_with_error_handling, datasets),
                total=len(datasets),
                desc="Processing datasets"
            ))


if __name__ == "__main__":
    run_configuration_analysis(num_processes=8)