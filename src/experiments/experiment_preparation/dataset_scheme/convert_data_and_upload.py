# Add these imports at the top
import json
import logging
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path

from tqdm import tqdm

from config.get_config import Config
from src.experiments.experiment_preparation.configuration_generation.TemplateVariationDimensions import TemplateVariationDimensions
from src.experiments.experiment_preparation.dataset_scheme.conversions.JSON_schema_validator import validate_json_data
from src.experiments.experiment_preparation.dataset_scheme.conversions.dataset_schema_adapter_huji import \
    convert_dataset
from src.experiments.experiment_preparation.dataset_scheme.hf_dataset_manager.dataset_publisher import \
    check_and_upload_parq_file_to_huggingface


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


def run_on_files(exp_dir: str, logger) -> list[Path]:
    """
    Create mapping between experiment_template files and template files
    Returns: Dict[template_number, (experiment_file_path, template_file_path)]
    """
    mapping = []

    # Get all instruction folders for the main progress bar
    instrcu_folders = os.listdir(exp_dir)

    for instrcu_folder in tqdm(instrcu_folders, desc="Processing instruction folders"):
        if not os.path.isdir(os.path.join(exp_dir, instrcu_folder)) or \
                instrcu_folder not in valid_paths:
            continue

        model_folders = os.listdir(os.path.join(exp_dir, instrcu_folder))
        for model_folder in tqdm(model_folders, desc=f"{instrcu_folder}", leave=False):
            if not os.path.isdir(os.path.join(exp_dir, instrcu_folder, model_folder)) or \
                    model_folder not in valid_paths[instrcu_folder]:
                continue
            if model_folder in ["Llama-2-7b-chat-hf",
                                "Mistral-7B-Instruct-v0.2",
                                "Phi-3-medium-4k-instruct"]:
                continue
            dataset_folders = os.listdir(os.path.join(exp_dir, instrcu_folder, model_folder))
            dataset_files = []
            file_name = model_folder.split("/")[-1]
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
                        exp_files = [os.path.join(system, file) for file in os.listdir(system)
                                     if (file.startswith('enumerator') or file.startswith('exp'))
                                     and file.endswith('.json')]
                        tqdm.write(f"Found {len(exp_files)} files in {model_folder}/{shot_number}")
                        dataset_files.extend(exp_files)
                        # return mapping
            mapping.append((file_name, dataset_files))
    return mapping


def convert_to_scheme_safe(items, config_params,
                           file_lock,
                           logger):
    start_time = time.time()
    try:
        convert_to_scheme(items, config_params,
                          file_lock,
                          logger)
    except Exception as e:
        file_name, dataset_files = items
        logger.error(f"Error processing file {file_name}: {e}")
        return None
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        file_name, dataset_files = items
        logger.info(f"Processing time for {file_name}: {elapsed_time:.2f} seconds")


def convert_to_scheme(items: list[Path], config_params, file_lock, logger: logging.Logger) -> None:
    all_data = []
    file_name, dataset_files = items
    try:
        for i, exp_file_path in enumerate(dataset_files):
            try:
                path_parts = str(exp_file_path).split(os.sep)
                model = path_parts[-5]
                dataset = path_parts[-4]
                name = path_parts[-1].split(".")[0]
                shot = next((part for part in path_parts if "shot" in part), "unknown")
                logger.info(f"Processing file for {model} - {shot}: {exp_file_path}")
                logger.info(
                    f"Processing file {i}/{len(dataset_files)} for {model} - {dataset} - {shot}: {exp_file_path}")
                with open(exp_file_path, 'r') as f:
                    input_data = json.load(f)
                logger.info(f"Reading file {exp_file_path}")
                template_name = input_data['template_name']
                from src.utils.Constants import Constants
                catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
                template_name.replace('MultipleChoiceTemplatesStructuredTopic',
                                      'MultipleChoiceTemplatesStructuredWithTopic')
                template_name.replace('MultipleChoiceTemplatesStructured',
                                      'MultipleChoiceTemplatesStructuredWithoutTopic')
                template_name = template_name.replace('MultipleChoiceTemplatesStructuredTopic',
                                                      'MultipleChoiceTemplatesStructuredWithTopic')
                template_name = template_name.replace('MultipleChoiceTemplatesStructured',
                                                      'MultipleChoiceTemplatesStructuredWithoutTopic')

                template_path = os.path.join(catalog_path, template_name + '.json')
                if not os.path.exists(template_path):
                    template_path = Path(
                        "/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/Catalog/MultipleChoiceTemplatesInstructionsWithoutTopic/enumerator_capitals_choicesSeparator_comma_shuffleChoices_False.json")
                with open(template_path, 'r') as f:
                    template_data = json.load(f)
                logger.info(f"Reading template file {template_path}")
                TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
                catalog_path = TemplatesGeneratorConstants.CATALOG_PATH
                from src.experiments.data_loading.CatalogManager import CatalogManager
                logger.info(f"Trying to load template {template_name} from {catalog_path}")
                catalog_manager = CatalogManager(catalog_path)
                template = catalog_manager.load_from_catalog(template_name)
                logger.info(f"Loaded template {template_name} from {catalog_path}")
                from src.experiments.data_loading.DatasetLoader import DatasetLoader
                logger.info(f"Reading template data {template_path}")

                llm_dataset_loader = DatasetLoader(card=input_data['card'],
                                                   template=template,
                                                   system_format=input_data['system_format'],
                                                   demos_taken_from='train' if input_data[
                                                                                   'num_demos'] < 1 else 'validation',
                                                   num_demos=input_data['num_demos'],
                                                   demos_pool_size=input_data['demos_pool_size'],
                                                   max_instances=input_data['max_instances'],
                                                   template_name=template_name, logger=logger)
                llm_dataset = llm_dataset_loader.load()
                dataset = llm_dataset.dataset
                # Read the schema file
                logger.info(f"Reading dataset for {model} - {shot}: {exp_file_path}")
                schema_file_path = r'/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/scheme.json'
                if not os.path.exists(schema_file_path):
                    schema_file_path = r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/scheme.json:66'
                with open(schema_file_path, 'r') as file:
                    schema_data = json.load(file)
                converted_data = convert_dataset(dataset, input_data, template_data)
                logger.info(f"Converting data for {model} - {shot}: {exp_file_path}")
                validate_json_data(converted_data, schema_data)
                logger.info(f"Validating data for {model} - {shot}: {exp_file_path}")
                all_data.extend(converted_data)
            except Exception as e:
                logger.error(f"Error processing file {exp_file_path}: {str(e)}")
                logger.info(f"Error processing file {exp_file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        logger.info(f"Error processing files: {str(e)}")
    REPO_NAME = "eliyahabba/llm-evaluation-dataset"  # Replace with your desired repo name
    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
    check_and_upload_parq_file_to_huggingface(all_data, repo_name=REPO_NAME, token=TOKEN, private=False,
                                              file_name=file_name, logger=logger)
    # After successful upload
    logger.info(f"Successfully processed and uploaded {file_name}")


def rename_files_parallel(file_mapping: list[Path], config_params: TemplateVariationDimensions,
                          logger: logging.Logger) -> None:
    num_processes = max(1, cpu_count() - 1)
    num_processes = max(1, 8)
    logger.info(f"Starting parallel processing with {num_processes} processes...")

    with Manager() as manager:
        file_lock = manager.Lock()
        process_func = partial(convert_to_scheme,
                               config_params=config_params,
                               file_lock=file_lock,
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

    experiment_dir = "/cs/labs/gabis/eliyahabba/results_with_good_names/"
    if not os.path.exists(experiment_dir):
        experiment_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/results"
    config_params = TemplateVariationDimensions()

    # Initialize tracker

    logger.info("Creating file mapping...")
    file_mapping = run_on_files(experiment_dir, logger)
    total_files = len(file_mapping)
    logger.info(f"Found {total_files} files to process")

    rename_files_parallel(file_mapping, config_params, logger)


if __name__ == "__main__":
    main()
