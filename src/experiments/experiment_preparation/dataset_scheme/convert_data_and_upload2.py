# Add these imports at the top
import json
import logging
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from config.get_config import Config
from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
from src.experiments.experiment_preparation.dataset_scheme.JSON_schema_validator import validate_json_data
from src.experiments.experiment_preparation.dataset_scheme.dataset_schema_adapter import convert_dataset
from src.experiments.experiment_preparation.dataset_scheme.hf_dataset_manager.dataset_publisher import \
    check_and_upload_to_huggingface


class ProcessTracker:
    def __init__(self):
        self.manager = Manager()
        self.processed_count = self.manager.Value('i', 0)
        self.uploaded_count = self.manager.Value('i', 0)
        self.failed_count = self.manager.Value('i', 0)
        self.model_stats = self.manager.dict()
        self.lock = self.manager.Lock()

    def increment_processed(self, model: str, shot: str):
        with self.lock:
            self.processed_count.value += 1
            if model not in self.model_stats:
                self.model_stats[model] = self.manager.dict()
            if shot not in self.model_stats[model]:
                self.model_stats[model][shot] = self.manager.Value('i', 0)
            self.model_stats[model][shot].value += 1

    def increment_uploaded(self):
        with self.lock:
            self.uploaded_count.value += 1

    def increment_failed(self):
        with self.lock:
            self.failed_count.value += 1

    def get_stats(self) -> Dict:
        with self.lock:
            stats = {
                'total_processed': self.processed_count.value,
                'total_uploaded': self.uploaded_count.value,
                'total_failed': self.failed_count.value,
                'model_stats': {
                    model: {shot: count.value for shot, count in shots.items()}
                    for model, shots in self.model_stats.items()
                }
            }
            return stats


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


valid_paths = {
    "MultipleChoiceTemplatesStructuredWithoutTopic": {
        "Mistral-7B-Instruct-v0.3": ["zero_shot", "three_shot"],
        "Mistral-7B-Instruct-v0.2": ["zero_shot"],
        "vicuna-7b-v1.5": ["zero_shot", "three_shot"],
        "gemma-7b-it": ["zero_shot", "three_shot"],
        "gemma-2b-it": ["zero_shot", "three_shot"],
        "Phi-3-mini-4k-instruct": ["zero_shot"],
        "Phi-3-medium-4k-instruct": ["zero_shot"],
        "OLMo-7B-Instruct-hf": ["three_shot"],
        "OLMo-1.7-7B-hf": ["zero_shot", "three_shot"],
        "Llama-2-13b-chat-hf": ["zero_shot"],
        "Llama-2-7b-hf": ["three_shot"],
        "Llama-2-7b-chat-hf": ["zero_shot"],
        "Meta-Llama-3-8B-Instruct": ["zero_shot"],
        "Llama-2-13b-hf": ["three_shot"]
    }
}


def run_on_files(exp_dir: str) -> list[Path]:
    """
    Create mapping between experiment_template files and template files
    Returns: Dict[template_number, (experiment_file_path, template_file_path)]
    """
    mapping = []

    # Get all instruction folders for the main progress bar
    instrcu_folders = os.listdir(exp_dir)

    for instrcu_folder in tqdm(instrcu_folders, desc="Processing instruction folders"):
        # בדיקה אם זו תיקייה והאם זו התיקייה הנכונה
        if not os.path.isdir(os.path.join(exp_dir, instrcu_folder)) or \
                instrcu_folder not in valid_paths:
            continue

        model_folders = os.listdir(os.path.join(exp_dir, instrcu_folder))
        for model_folder in tqdm(model_folders, desc=f"{instrcu_folder}", leave=False):
            if not os.path.isdir(os.path.join(exp_dir, instrcu_folder, model_folder)) or \
                    model_folder not in valid_paths[instrcu_folder]:
                continue

            dataset_folders = os.listdir(os.path.join(exp_dir, instrcu_folder, model_folder))
            for dataset_folder in dataset_folders:
                if not os.path.isdir(os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder)):
                    continue

                shot_numbers = os.listdir(os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder))
                for shot_number in shot_numbers:
                    if not os.path.isdir(
                            os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number)) or \
                            shot_number not in valid_paths[instrcu_folder][model_folder]:
                        continue

                    shots = os.path.join(exp_dir, instrcu_folder, model_folder, dataset_folder, shot_number)
                    system_folders = os.listdir(shots)
                    for system_folder in system_folders:
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

                        if exp_files:  # Print info when files are found
                            tqdm.write(f"Found {len(exp_files)} files in {model_folder}/{shot_number}")

    return mapping


def convert_to_scheme(item: Path, config_params, file_lock, tracker: ProcessTracker, logger: logging.Logger) -> None:
    exp_file_path = item
    try:
        # Extract model and shot from path
        path_parts = str(exp_file_path).split(os.sep)
        model = next((part for part in path_parts if part in valid_paths["MultipleChoiceTemplatesStructured"]),
                     "unknown")
        shot = next((part for part in path_parts if "shot" in part), "unknown")

        logger.info(f"Processing file for {model} - {shot}: {exp_file_path}")

        with open(exp_file_path, 'r') as f:
            input_data = json.load(f)

        template_name = input_data['template_name']
        from src.utils.Constants import Constants
        catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
        template_name.replace('MultipleChoiceTemplatesStructuredTopic',
                              'MultipleChoiceTemplatesStructuredWithTopic')
        template_name.replace('MultipleChoiceTemplatesStructured', 'MultipleChoiceTemplatesStructuredWithoutTopic')
        template_name = template_name.replace('MultipleChoiceTemplatesStructuredTopic',
                                              'MultipleChoiceTemplatesStructuredWithTopic')
        template_name = template_name.replace('MultipleChoiceTemplatesStructured',
                                              'MultipleChoiceTemplatesStructuredWithoutTopic')

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
                                           demos_taken_from='train' if input_data[
                                                                           'num_demos'] < 1 else 'validation',
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

        check_and_upload_to_huggingface(converted_data, repo_name=REPO_NAME, token=TOKEN, private=False)
        # After successful upload
        tracker.increment_processed(model, shot)
        tracker.increment_uploaded()
        logger.info(f"Successfully processed and uploaded {exp_file_path}")

    except Exception as e:
        tracker.increment_failed()
        logger.error(f"Error processing file {exp_file_path}: {str(e)}")


def rename_files_parallel(file_mapping: list[Path], config_params: ConfigParams, tracker: ProcessTracker,
                          logger: logging.Logger) -> None:
    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Starting parallel processing with {num_processes} processes...")

    with Manager() as manager:
        file_lock = manager.Lock()
        process_func = partial(convert_to_scheme,
                               config_params=config_params,
                               file_lock=file_lock,
                               tracker=tracker,
                               logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, file_mapping),
                total=len(file_mapping),
                desc="Processing files"
            ))


def main():
    logger = setup_logging()
    logger.info("Starting processing...")

    experiment_dir = "/cs/labs/gabis/eliyahabba/results"
    config_params = ConfigParams()

    # Initialize tracker
    tracker = ProcessTracker()

    logger.info("Creating file mapping...")
    file_mapping = run_on_files(experiment_dir)
    total_files = len(file_mapping)
    logger.info(f"Found {total_files} files to process")

    rename_files_parallel(file_mapping, config_params, tracker, logger)

    # Print final statistics
    stats = tracker.get_stats()
    logger.info("\nProcessing Summary:")
    logger.info(f"Total files processed: {stats['total_processed']}/{total_files}")
    logger.info(f"Successfully uploaded: {stats['total_uploaded']}")
    logger.info(f"Failed: {stats['total_failed']}")

    logger.info("\nBreakdown by model and shot:")
    for model, shots in stats['model_stats'].items():
        logger.info(f"\n{model}:")
        for shot, count in shots.items():
            logger.info(f"  {shot}: {count} files")


if __name__ == "__main__":
    main()
