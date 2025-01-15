import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from typing import Dict, Optional, Iterator, Union, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
import requests
from datasets import load_dataset
from tqdm import tqdm

from src.experiments.experiment_preparation.dataset_scheme.conversions.RunOutputMerger import RunOutputMerger
from src.experiments.experiment_preparation.dataset_scheme.conversions.download_ibm_data import generate_file_names
from src.experiments.experiment_preparation.datasets_configurations.DatasetConfigFactory import DatasetConfigFactory
from src.utils.Constants import Constants


def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'processing_log_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class RunOutputMerger:
    """A class to process and merge run data with associated metadata."""

    def __init__(self, parquet_path: Path, batch_size: int = 1000):
        """
        Initialize the RunDataProcessor.

        Args:
            parquet_path (str): Path to the parquet file
            batch_size (int): Size of batches for processing
        """
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.runs_df = self._load_runs_dataset()

    def _load_runs_dataset(self) -> pd.DataFrame:
        """
        Load and process the runs dataset into a dictionary for O(1) lookups.

        Returns:
            Dict[str, Dict[str, Any]]: Processed runs dictionary
        """
        runs_ds = load_dataset("OfirArviv/HujiCollabRun")
        train_runs_ds = runs_ds['run']
        df = train_runs_ds.to_pandas()
        return df

    def _add_run_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add run-related columns to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with added run columns
        """

        # Map run information to DataFrame
        run_df_unique = self.runs_df.drop_duplicates(subset=['run_id'])
        run_df_unique.columns = ['run_' + col if col != 'run_id' else col for col in run_df_unique.columns]
        keep_columns = ['run_id', 'run_model_args', 'run_unitxt_recipe']
        run_df_unique = run_df_unique[keep_columns]

        merged_df = df.merge(
            run_df_unique,
            on='run_id',
            how='left'
        )

        return merged_df

    def process_batches(self) -> Iterator[pd.DataFrame]:
        """
        Process the parquet file in batches, adding run information.

        Yields:
            pd.DataFrame: Processed batch with added run information
        """
        try:
            parquet_file = pq.ParquetFile(self.parquet_path)
            columns_to_keep = ['run_id', 'scores', 'references', 'generated_text', 'cumulative_logprob','raw_input']

            # for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=columns_to_keep):
                df = batch.to_pandas()
                # eval on the raw_input column, and next take only task_data key
                df['question'] = df['raw_input'].apply(lambda x: eval(x)['task_data']).apply(lambda x: eval(x)['question'
                if 'question' in x else 'context'])
                # drop the raw_input column
                df.drop(columns=['raw_input'], inplace=True)
                df = self._add_run_columns(df)
                yield df
            # eval(df.iloc[0]['run_model_args'])['model']'scores',['run_unitxt_recipe']'card','generated_text'
        except Exception as e:
            print(f"Error processing parquet file: {str(e)}")
            raise


class Converter:
    def __init__(self, models_metadata_path: Union[str, Path]):
        """Initialize converter with path to models metadata."""
        self.models_metadata_path = Path(models_metadata_path)
        self._load_models_metadata()
        self._template_cache = {}  # Cache for templates
        self._index_map_cache = {}  # Cache for index map
    def _load_models_metadata(self) -> None:
        """Load models metadata from file."""
        with open(self.models_metadata_path) as f:
            self.models_metadata = json.load(f)

    def _parse_config_string(self, config_string: str) -> Dict:
        """Parse configuration string to dictionary."""
        return dict(pair.split('=') for pair in config_string.split(','))

    def _build_model_section(self, row: pd.Series
                             ) -> Tuple[str, Optional[str]]:
        """Build model section of schema."""

        # Get model configuration
        model_args = json.loads(row['run_model_args'])
        model_metadata = self.models_metadata[model_args['model']]
        return model_args['model'], model_metadata.get("ModelInfo", {}).get("family")

    def parse_template_params(self, recipe: Dict) -> Tuple[str, str, dict]:
        """Extract template parameters from recipe configuration."""
        config = recipe['template'].split('.')[-1]

        enumerator = config.split('enumerator_')[1].split("_")[0]
        choice_sep = config.split('choicesSeparator_')[1].split("_")[0]
        shuffle = config.split('shuffleChoices_')[1]

        choices_order_map = {"randiom": {
            "method": "random",
            "description": "Randomly shuffles all choices"
        },
            "False":
                {
                    "method": "none",
                    "description": "Preserves the original order of choices as provided"
                },
            "lengthSort":
                {
                    "method": "shortest_to_longest",
                    "description": "Orders choices by length, from shortest to longest text"
                },
            "lengthSortReverse":
                {
                    "method": "longest_to_shortest",
                    "description": "Orders choices by length, from longest to shortest text"
                },
            "placeCorrectChoiceFirst":
                {
                    "method": "correct_first",
                    "description": "Places the correct answer as the first choice"
                },
            "placeCorrectChoiceFourth":
                {
                    "method": "correct_last",
                    "description": "Places the correct answer as the last choice"
                },
            "alphabeticalSort":
                {
                    "method": "alphabetical",
                    "description": "Orders choices alphabetically (A to Z)"
                },
            "alphabeticalSortReverse":
                {
                    "method": "reverse_alphabetical",
                    "description": "Orders choices in reverse alphabetical order (Z to A)"
                }
        }
        choices_order = choices_order_map[shuffle]

        class SeparatorMap:
            SEPARATOR_MAP = {
                "space": "\\s",
                " ": "\\s",
                "newline": "\n",
                "comma": ", ",
                "semicolon": "; ",
                "pipe": " | ",
                "OR": " OR ",
                "or": " or ",
                "orLower": " or ",
                "OrCapital": " OR "
            }

            @classmethod
            def get_separator(cls, separator: str) -> str:
                return cls.SEPARATOR_MAP.get(separator, separator)

        separator = SeparatorMap.get_separator(choice_sep)
        return enumerator, separator, choices_order

    def _get_template(self, run_unitxt_recipe: dict, logger=None) -> str:
        template_name = run_unitxt_recipe['template'].split('templates.huji_workshop.')[-1]
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
        template_name = template_name.replace(".", "/")
        template_path = Path(catalog_path, template_name + '.json')

        with open(template_path, 'r') as f:
            template_data = json.load(f)

        self._template_cache[template_name] = template_data['input_format']
        prompt_paraphrasing = self.get_prompt_paraphrasing(run_unitxt_recipe['card'])
        input_format = template_data['input_format']
        for prompt in prompt_paraphrasing.get_all_prompts():
            if prompt.text == input_format:
                return prompt.name
        raise ValueError(f"Template {template_name} not found in catalog")

    def _build_prompt_section(self, recipe: Dict,
                              logger=None) -> Dict:
        """Build prompt configuration section of schema."""
        # Process demonstration examples
        enumerator, separator, choices_order = self.parse_template_params(recipe)
        return {
            "template": self._get_template(recipe, logger),
            "separator": separator,
            "enumerator": enumerator,
            "choices_order": choices_order['method'],
            "shots": int(recipe['num_demos']),
        }

    def _build_output_section(self, row: pd.Series) -> Tuple[str, float]:
        """Build output section of schema."""
        return row['generated_text'], row['cumulative_logprob']

    def _build_evaluation_section(self, row: pd.Series) -> Dict:
        """Build evaluation section of schema."""
        ground_truth = eval(row['references'])[0]
        score = eval(row['scores'])['score']
        return ground_truth, score

    def get_prompt_paraphrasing(self, dataset):
        if "mmlu_pro" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("MMLU_PRO")
        elif "mmlu" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("MMLU")
        elif "ai2_arc" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("AI2_ARC")
        elif "hellaswag" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("HellaSwag")
        elif "openbook_qa" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("OpenBookQA")
        elif "social_iqa" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("Social_IQa")
        else:
            raise ValueError(f"Dataset {dataset} not found in DatasetConfigFactory")

    def build_instance_section(self, row: pd.Series, recipe) -> Dict:
        map_file_name = recipe['card'].split('.')[1]
        if map_file_name == "ai2_arc":
            map_file_name = recipe['card'].split('.')[1]
        current_dir = Path(__file__).parents[1] / "conversions"
        map_file_path = current_dir / "hf_map_data" / f"{map_file_name}_samples.json"
        if map_file_name in self._index_map_cache:
            index_map = self._index_map_cache[map_file_name]
        else:
            with open(map_file_path, 'r') as file:
                index_map = json.load(file)
            self._index_map_cache[map_file_name] = index_map
        return index_map[row['question']]['index']


    def convert_dataframe(self, df: pd.DataFrame):
        """Convert entire DataFrame using vectorized operations."""
        # Get recipes
        recipes = df['run_unitxt_recipe'].apply(self._parse_config_string)
        index_map = df.apply(
            lambda row: self.build_instance_section(row, self._parse_config_string(row['run_unitxt_recipe'])),
            axis=1).rename('index')

        # Build all sections in parallel
        model_sections = df.apply(self._build_model_section, axis=1, result_type='expand') \
            .rename(columns={0: 'model', 1: 'family'})

        prompt_sections = recipes.apply(lambda x: pd.Series(self._build_prompt_section(x)))

        output_sections = df.apply(self._build_output_section, axis=1, result_type='expand') \
            .rename(columns={0: 'generated_text', 1: 'cumulative_logprob'})

        eval_sections = df.apply(self._build_evaluation_section, axis=1, result_type='expand') \
            .rename(columns={0: 'ground_truth', 1: 'score'})

        dataset = recipes.str['card'].str.split('.').str[1:].str.join('.').rename('dataset')



        # Combine all sections
        return pd.concat([
            index_map,
            model_sections,
            dataset,
            prompt_sections,
            output_sections,
            eval_sections
        ], axis=1)

        # Add prompt config columns
        for key in ['template', 'separator', 'enumerator', 'choices_order', 'shots']:
            result_data[key] = prompt_configs.apply(lambda x: x[key])

        return pd.DataFrame(result_data)


def main(file_path: Path = Path(f"~/Downloads/data_2025-01.parquet"),
         process_output_dir: Path = Path(f"~/Downloads/data_2025-01.parquet"),
         logger=None):
    """Main function to demonstrate usage."""
    parquet_path = file_path
    logger.info(f"Processing file: {parquet_path}")
    processor = RunOutputMerger(parquet_path)
    num_of_batches = 0
    models_metadata_path = Path(Constants.ExperimentConstants.MODELS_METADATA_PATH)
    converter = Converter(models_metadata_path)

    # Create output path
    output_path = os.path.join(process_output_dir,
                               f"processed_{parquet_path.name}")
    if os.path.exists(output_path):
        logger.info(f"Output file already exists: {output_path}")
        return
    logger.info(f"Output path: {output_path}")
    # Process first batch to get schema
    first_batch = next(processor.process_batches())
    data = converter.convert_dataframe(first_batch)

    # Convert to PyArrow table and get schema
    table = pa.Table.from_pandas(data)

    # Create ParquetWriter with the schema
    writer = pq.ParquetWriter(
        output_path,
        schema=table.schema,
    )

    # Write first batch
    writer.write_table(table)
    num_of_batches = 1
    logger.info(f"Batches processed: {num_of_batches}")

    # Process remaining batches
    for processed_batch in processor.process_batches():
        num_of_batches += 1
        logger.info(f"Batches processed: {num_of_batches}")

        # Convert the batch
        data = converter.convert_dataframe(processed_batch)
        table = pa.Table.from_pandas(data)

        # Write batch
        writer.write_table(table)

    # Close the writer
    writer.close()
    logger.info(f"Processed {num_of_batches} batches.")
    logger.info(f"Output saved to: {output_path}")


def download_huggingface_files_parllel(output_dir: Path, process_output_dir, urls, scheme_files_dir):
    """
    Downloads parquet files from Hugging Face to a specified directory
/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/analsyis/save_data_for_analysis.py
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_processes = min(24, cpu_count() - 1)
    logger = setup_logging()
    logger.info("Starting processing...")
    logger.info(f"Starting parallel processing with {num_processes} processes...")

    with Manager() as manager:
        process_func = partial(process_file_safe, output_dir=output_dir, process_output_dir=process_output_dir,
                               scheme_files_dir=scheme_files_dir, logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, urls),
                total=len(urls),
                desc="Processing files"
            ))


def process_file_safe(url, output_dir: Path, process_output_dir, scheme_files_dir, logger):
    pid = os.getpid()
    start_time = time.time()
    try:
        logger.info(f"Process {pid} starting to process file {url}")
        process_file(url, output_dir=output_dir, process_output_dir=process_output_dir,
                     scheme_files_dir=scheme_files_dir,
                     logger=logger)
    except Exception as e:
        logger.error(f"Process {pid} encountered error processing file {url}: {e}")
        return None
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Process {pid} finished processing {url}. Processing time: {elapsed_time:.2f} seconds")


def process_file(url, output_dir: Path, process_output_dir: Path, scheme_files_dir, logger):
    # Extract date from URL for the output filename
    # Example: from URL get '2024-12-16' for the filename
    original_filename = url.split('/')[-1]
    output_path = output_dir / original_filename
    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
    else:
        try:
            logger.info(f"Downloading {original_filename}...")

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name

                response = requests.get(url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            shutil.move(temp_path, output_path)

            logger.info(f"Download completed: {original_filename}")
            print(f"Download completed: {original_filename}")

        except (requests.exceptions.RequestException, IOError) as e:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            print(f"Error downloading {original_filename}: {e}")
            logger.error(f"Error downloading {original_filename}: {e}")
    main(output_path, process_output_dir, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', nargs='+', help='List of urls to download')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing parquet files', default=1000)
    parser.add_argument('--repo_name', help='Repository name for the schema files')
    parser.add_argument('--scheme_files_dir', help='Directory to store the scheme files')
    args = parser.parse_args()
    args.output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full"
    process_output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed"
    if not os.path.exists(process_output_dir):
        process_output_dir = "/Users/ehabba/ibm_results_data_full_processed"
        os.makedirs(process_output_dir, exist_ok=True)

    # parquet_path = Path("~/Downloads/data_sample.parquet")
    # convert_to_scheme_format(parquet_path, batch_size=100)
    main_path = 'https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/'
    # List of URLs to download
    start_time = datetime(2025, 1, 11, 19, 0, tzinfo=pytz.UTC)

    end_time = datetime(2025, 1, 12, 19, 0, tzinfo=pytz.UTC)

    files = generate_file_names(start_time, end_time)
    args.urls = [f"{main_path}{file}" for file in files]
    args.urls = args.urls + [
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2025-01-09T19%3A00%3A00%2B00%3A00_2025-01-09T23%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2025-01-10T14%3A00%3A00%2B00%3A00_2025-01-11T18%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2025-01-10T00%3A00%3A00%2B00%3A00_2025-01-10T10%3A00%3A00%2B00%3A00.parquet",
        "https://huggingface.co/datasets/OfirArviv/HujiCollabOutput/resolve/main/data_2025-01-10T11%3A00%3A00%2B00%3A00_2025-01-10T13%3A00%3A00%2B00%3A00.parquet",
    ]
    # args.output_dir = "/Users/ehabba/Downloads/"
    # args.urls = ["tmp/data_2025-01.parquet"]
    # process_output_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/conversions/process_output_dir"
    # os.makedirs(process_output_dir, exist_ok=True)
    download_huggingface_files_parllel(output_dir=Path(args.output_dir), process_output_dir=process_output_dir,
                                       urls=args.urls,
                                       scheme_files_dir=args.scheme_files_dir)
