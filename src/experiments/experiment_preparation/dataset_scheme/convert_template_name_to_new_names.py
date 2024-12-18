import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from config.get_config import Config
from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
from src.experiments.experiment_preparation.dataset_scheme.JSON_schema_validator import validate_json_data
from src.experiments.experiment_preparation.dataset_scheme.dataset_schema_adapter import convert_dataset
from src.experiments.experiment_preparation.dataset_scheme.hf_dataset_manager.dataset_publisher import \
    upload_to_huggingface
from src.utils.Constants import Constants


def run_on_files(exp_dir: str) -> list[Path]:
    """
    Create mapping between experiment_template files and template files
    Returns: Dict[template_number, (experiment_file_path, template_file_path)]
    """
    mapping = []
    for instrcu_folder in os.listdir(exp_dir):
        # if not folder continue
        if not os.path.isdir(os.path.join(exp_dir, instrcu_folder)):
            continue
        for model_folder in os.listdir(os.path.join(exp_dir, instrcu_folder)):
            if not os.path.isdir(os.path.join(exp_dir, instrcu_folder, model_folder)):
                continue
            for dataset_folder in os.listdir(os.path.join(exp_dir, instrcu_folder, model_folder)):
                if not os.path.isdir(os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder)):
                    continue
                for shot_number in os.listdir(os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder)):
                    if not os.path.isdir(
                            os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number)):
                        continue
                    shots = os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number)
                    for system_folder in os.listdir(shots):
                        if not os.path.isdir(
                                os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number,
                                             system_folder)):
                            continue
                        system = os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number,
                                              system_folder)
                        exp_files = [os.path.join(system, file) for file in os.listdir(system)
                                     if file.startswith('enumerator') and file.endswith('.json')]
                        # add to mapping
                        for exp_file in exp_files:
                            exp_path = Path(exp_file)
                            mapping.append(exp_path)
                            return mapping
                            break

    return mapping


def convert_to_scheme(item: Path, config_params, file_lock) -> None:
    """Process a single file rename operation with thread-safe checking"""
    exp_file_path = item
    try:
        with open(exp_file_path, 'r') as f:
            input_data = json.load(f)
        template_name=  input_data['template_name']
        from src.utils.Constants import Constants
        catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
        template_name.replace('MultipleChoiceTemplatesStructuredTopic', 'MultipleChoiceTemplatesStructuredWithTopic')
        template_name.replace('MultipleChoiceTemplatesStructured', 'MultipleChoiceTemplatesStructuredWithoutTopic')
        template_name = template_name.replace('MultipleChoiceTemplatesStructuredTopic', 'MultipleChoiceTemplatesStructuredWithTopic')
        template_name = template_name.replace('MultipleChoiceTemplatesStructured', 'MultipleChoiceTemplatesStructuredWithoutTopic')

        template_path = os.path.join(catalog_path, template_name + '.json')
        with open(template_path, 'r') as f:
            template_data = json.load(f)

        TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
        catalog_path = TemplatesGeneratorConstants.CATALOG_PATH
        from src.experiments.data_loading.CatalogManager import CatalogManager
        catalog_manager = CatalogManager(catalog_path)
        template = catalog_manager.load_from_catalog(template_name)
        from src.experiments.data_loading.DatasetLoader import DatasetLoader

        llm_dataset_loader = DatasetLoader(card=input_data['card'],
                                           template=template,
                                           system_format=input_data['system_format'],
                                           demos_taken_from='train' if input_data['num_demos']<1 else 'validation',
                                           num_demos=input_data['num_demos'],
                                           demos_pool_size=input_data['demos_pool_size'],
                                           max_instances=input_data['max_instances'],
                                           template_name=template_name)
        llm_dataset = llm_dataset_loader.load()
        dataset = llm_dataset.dataset
        # Read the schema file
        schema_file_path = r'/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/scheme.json'
        with open(schema_file_path, 'r') as file:
            schema_data = json.load(file)
        converted_data = convert_dataset(dataset, input_data, template_data)
        validate_json_data(converted_data, schema_data)
        REPO_NAME = "eliyahabba/llm-evaluation-dataset"  # Replace with your desired repo name
        config = Config()
        TOKEN = config.config_values.get("hf_access_token", "")

        upload_to_huggingface(converted_data, repo_name=REPO_NAME, token=TOKEN, private=False)
        # print(converted_data)
    except Exception as e:
        print(f"Error processing file {exp_file_path}: {str(e)}")


def rename_files_parallel(file_mapping: list[Path], config_params: ConfigParams) -> None:
    """Parallel version of rename_files using multiprocessing with thread-safe file operations"""
    # Determine number of processes to use (leave one core free)
    num_processes = max(1, cpu_count() - 1)

    print(f"Starting parallel renaming with {num_processes} processes...")

    # Create a manager for sharing the lock between processes
    with Manager() as manager:
        # Create a lock for file operations
        file_lock = manager.Lock()

        # Create partial function with config_params and lock
        process_func = partial(convert_to_scheme,
                               config_params=config_params,
                               file_lock=file_lock)

        # Create process pool and map the work
        with Pool(processes=num_processes) as pool:
            # Convert the dictionary items to a list for tqdm
            items = file_mapping

            # Use imap_unordered for better performance with progress bar
            list(tqdm(
                pool.imap_unordered(process_func, items),
                total=len(items),
                desc="Renaming files"
            ))


# Modified main function to use parallel version
def main():
    experiment_dir = "/cs/labs/gabis/eliyahabba/LLM-Evaluation/results"

    config_params = ConfigParams()

    print("Creating file mapping...")
    file_mapping = run_on_files(experiment_dir)

    print(f"\nFound {len(file_mapping)} matching file pairs")
    # split the file to 2 parts
#    file_mapping1 = {k: v for i, (k, v) in enumerate(file_mapping.items()) if i % 2 == 0}
#    file_mapping2 = {k: v for i, (k, v) in enumerate(file_mapping.items()) if i % 2 == 1}
    print(f"\nFound {len(file_mapping)} matching file pairs")

    print("\nRenaming files in parallel...")
    rename_files_parallel(file_mapping, config_params)

    print("\nFinished processing all files.")


if __name__ == "__main__":
    main()

