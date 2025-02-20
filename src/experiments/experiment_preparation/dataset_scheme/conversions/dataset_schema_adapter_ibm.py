import hashlib
import json
import re
from dataclasses import dataclass
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.experiments.experiment_preparation.dataset_scheme.conversions.RunOutputMerger import RunOutputMerger
from src.experiments.experiment_preparation.datasets_configurations.DatasetConfigFactory import DatasetConfigFactory
from src.utils.Constants import Constants


class QuantizationPrecision(str, Enum):
    """Supported quantization precisions."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


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


@dataclass
class LogProbToken:
    """Structure for token-level log probability information."""
    logprob: float
    rank: int
    decoded_token: str

    @classmethod
    def from_dict(cls, data: Dict) -> 'LogProbToken':
        return cls(
            logprob=data["logprob"],
            rank=data["rank"],
            decoded_token=data["decoded_token"]
        )

    def to_dict(self) -> Dict:
        return {
            "logprob": self.logprob,
            "rank": self.rank,
            "decoded_token": self.decoded_token
        }


class SchemaConverter:
    """Handles conversion of model evaluation data to standardized schema format."""

    def __init__(self, models_metadata_path: Union[str, Path], logger=None):
        """
        Initialize converter with path to models metadata.
        
        Args:
            models_metadata_path: Path to models metadata file
            logger: Logger instance to use (optional)
        """
        self.models_metadata_path = Path(models_metadata_path)
        self._load_models_metadata()
        self._template_cache = {}  # Cache for templates
        self._index_map_cache = {}  # Cache for index map
        self.logger = logger

    def set_logger(self, logger):
        """Set logger after initialization."""
        self.logger = logger

    def _load_models_metadata(self) -> None:
        """Load models metadata from file."""
        with open(self.models_metadata_path) as f:
            self.models_metadata = json.load(f)

    def add_token_index_to_tokens(self, log_probs_array: np.ndarray) -> List[Dict]:
        """Transform log probabilities to schema format."""
        return log_probs_array.tolist()

    def transform_demos(self, demos: List[Dict]) -> List[Dict]:
        """Transform demonstration examples to schema format."""
        transformed = []
        for demo in demos:
            choices = [
                {"id": opt.split(".")[0], "text": choice}
                for opt, choice in zip(demo['options'], demo['choices'])
            ]
            question_key = 'question' if 'question' in demo else 'context'

            transformed_demo = {
                "topic": None if 'topic' not in demo else demo['topic'],
                "question": demo[question_key],
                "choices": choices,
                "answer": demo["answer"]
            }
            transformed.append(transformed_demo)
        return transformed

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

        separator = SeparatorMap.get_separator(choice_sep)
        return enumerator, separator, choices_order

    def parse_generation_args(self, args: dict) -> Dict:
        """Parse generation arguments from string format."""
        return {
            "use_vllm": True,
            "temperature": args.get("temperature"),
            "top_p": args.get("top_p"),
            "top_k": args.get("top_k"),
            "max_tokens": args.get("max_tokens"),
            "stop_sequences": args.get("stop_sequences", [])
        }

    def convert_quantization(self, value: str) -> str:
        """Convert quantization value to schema format."""
        mapping = {
            "None": QuantizationPrecision.NONE,
            "half": QuantizationPrecision.FLOAT16,
            "float16": QuantizationPrecision.FLOAT16,
            "float32": QuantizationPrecision.FLOAT32,
            "int8": QuantizationPrecision.INT8,
            "int4": QuantizationPrecision.INT4
        }
        if value not in mapping:
            raise ValueError(f"Unsupported quantization value: {value}")
        return mapping[value].value

    def calculate_perplexity(self, logprobs: List) -> float:
        """Calculate perplexity from log probabilities."""
        if logprobs[0] is None:
            logprobs = logprobs[1:]
        logprob_values = np.array([item[0]['logprob'] for item in logprobs])
        return np.exp(-np.mean(logprob_values))

    def convert_row(self, row: pd.Series, probs: bool = True, logger: Optional = None) -> Dict:
        """Convert a single DataFrame row to schema format."""
        try:
            task_data = row['task_data']
            if logger:
                logger.debug(f"Task data: {task_data}")
            
            recipe = self._parse_config_string(row['run_unitxt_recipe'])
            if logger:
                logger.debug(f"Recipe: {recipe}")
            
            combined_strings = row['run_id'] + row['id']
            evaluation_id = hashlib.sha256(combined_strings.encode()).hexdigest()

            model_section = self._build_model_section(row)
            prompt_section = self._build_prompt_section(task_data, recipe, logger=logger)
            instance_section = self._build_instance_section(row, task_data, recipe, probs)
            output_section = self._build_output_section(row, probs)
            evaluation_section = self._build_evaluation_section(task_data, row)
            
            schema = {
                "evaluation_id": evaluation_id,
                "model": model_section,
                "prompt_config": prompt_section,
                "instance": instance_section,
                "output": output_section,
                "evaluation": evaluation_section
            }

            return schema
        
        except Exception as e:
            if logger:
                logger.error(f"Error in convert_row: {str(e)}")
                logger.error(f"Row data: {row.to_dict()}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _parse_config_string(self, config_string: str) -> Dict:
        """Parse configuration string to dictionary."""
        return dict(pair.split('=') for pair in config_string.split(','))

    def get_prompt_paraphrasing(self, card_name: str):
        """Get prompt paraphrasing from dataset config."""
        # Get dataset name from card name (e.g., "cards.mmlu" -> "mmlu")
        dataset = card_name.split("cards.")[1]

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
        elif "race" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("Race")
        elif "quality" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("QuALITY")
        else:
            raise ValueError(f"Dataset {dataset} not found in DatasetConfigFactory")

    def _get_template(self, recipe: Dict, logger=None) -> Dict:
        """Get template information from recipe."""
        template_name = recipe['template'].split('templates.huji_workshop.')[-1]
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
        template_name = template_name.replace(".", "/")
        template_path = Path(catalog_path, template_name + '.json')

        with open(template_path, 'r') as f:
            template_data = json.load(f)

        # Get both template text and name
        template_text = template_data['input_format']
        prompt_paraphrasing = self.get_prompt_paraphrasing(recipe['card'])

        # Find matching prompt to get its name
        for prompt in prompt_paraphrasing.get_all_prompts():
            if prompt.text == template_text:
                template_info = {
                    "text": prompt.text,  # The actual instruction text
                    "name": prompt.name  # The name/identifier of the instruction
                }
                self._template_cache[template_name] = template_info
                return template_info

        raise ValueError(f"Template {template_name} not found in catalog")

    def _build_model_section(self, row: pd.Series
                             ) -> Dict:
        """Build model section of schema."""

        # Get model configuration
        model_args = row['run_model_args']
        model_metadata = self.models_metadata[model_args['model']]
        return {
            "model_info": {
                "name": model_args['model'],
                "family": model_metadata.get("ModelInfo", {}).get("family")
            },
            "configuration": {
                "architecture": model_metadata.get("Configuration", {}).get("architecture"),
                "context_window": model_metadata['Configuration']['context_window'],
                "is_instruct": model_metadata['Configuration'].get('is_instruct', False),
                "parameters": model_metadata['Configuration'].get('parameters'),
                "hf_path": model_args['model'],
                "revision": model_metadata.get("Configuration", {}).get("revision")
            },
            "inference_settings": {
                "quantization": {
                    "bit_precision": self.convert_quantization(
                        row['run_quantization_bit_count']
                    ),
                    "method": row['run_quantization_type']
                },
                "generation_args": self.parse_generation_args(
                    row['run_generation_args']
                )
            }
        }

    def _build_prompt_section(self, task_data: Dict, recipe: Dict, logger=None) -> Dict:
        """Build prompt configuration section of schema."""
        # Process demonstration examples
        enumerator, separator, choices_order = self.parse_template_params(recipe)
        demos = ([] if 'demos' not in task_data else
                 self.transform_demos(task_data['demos']))

        template_info = self._get_template(recipe, logger)

        return {
            "prompt_class": "MultipleChoice",
            "dimensions": {
                "instruction_phrasing": {
                    "text": template_info["text"],
                    "name": template_info["name"]
                },
                "separator": separator,
                "enumerator": enumerator,
                "choices_order": choices_order,
                "shots": int(recipe['num_demos']),
                "demonstrations": demos
            }
        }

    def _build_instance_section(self, row: pd.Series, task_data: Dict,
                                recipe: Dict, probs: bool = True
                                ) -> Dict:
        """Build instance section of schema."""
        # Extract choice information
        choices = [
            {"id": opt.split(".")[0], "text": choice}
            for opt, choice in zip(task_data['options'], task_data['choices'])
        ]

        # Transform log probabilities
        if probs:
            prompt_logprobs = row['prompt_logprobs']

            prompt_tokens_logprobs = [
                self.add_token_index_to_tokens(logprob_dict['log_probs'])
                for logprob_dict in prompt_logprobs
            ]
            perplexity = self.calculate_perplexity(prompt_tokens_logprobs)
        else:
            prompt_tokens_logprobs = []
            perplexity = None

        map_file_name = recipe['card'].split('.')[1]
        if map_file_name == "ai2_arc":
            map_file_name = recipe['card'].split('.')[1]

        if map_file_name.startswith("global_mmlu"):
            # add also the language in the second part of the string
            map_file_name = f"{map_file_name}.{recipe['card'].split('.')[2]}"
        current_dir = Path(__file__).parents[2]
        map_file_path = current_dir / "dataset_scheme/conversions/hf_map_data/" / f"{map_file_name}_samples.parquet"
        if map_file_name in self._index_map_cache:
            index_map = self._index_map_cache[map_file_name]
        else:
            index_map = pd.read_parquet(map_file_path)
            self._index_map_cache[map_file_name] = index_map

        hf_repo = f"cais/{recipe['card'].split('.')[1]}" if map_file_name == "mmlu" else f"Rowan/hellaswag" if map_file_name == "hellaswag" else f"allenai/{map_file_name}"

        card = map_file_name
        if card == "mmlu":
            hf_repo = f"cais/mmlu"
        elif card == "mmlu_pro":
            hf_repo = "TIGER-Lab/MMLU-Pro"
        elif card == "hellaswag":
            hf_repo = "Rowan/hellaswag"
        elif card == "openbookqa":
            hf_repo = "allenai/openbookqa"
        elif card == "social_i_qa":
            hf_repo = "allenai/social_i_qa"
        elif "ai2_arc" in card:
            if "ARC-Challenge" in card:
                hf_repo = "allenai/ai2_arc/ARC-Challenge"
            elif "ARC-Easy" in card:
                hf_repo = "allenai/ai2_arc/ARC-Easy"
        elif "quaily" in card:
            hf_repo = "emozilla/quality"
        elif "global_mmlu_lite" in card:
            hf_repo = "CohereForAI/Global-MMLU-Lite"
        elif "global_mmlu" in card:
            hf_repo = "CohereForAI/Global-MMLU"
        else:
            hf_repo = None
        question_key = 'question' if 'question' in task_data else 'context'

        hf_index, hf_split = self._get_guestion_index(index_map, task_data)

        # Determine language from dataset name
        dataset_name = recipe['card'].split("cards.")[1]
        lang_match = re.match(r'(global_mmlu|global_mmlu_lite_cs|global_mmlu_lite_ca)\.(\w+)$', dataset_name)
        language = lang_match.group(2).lower() if lang_match else "en"

        return {
            "task_type": "classification",
            "raw_input": row['prompt'][-1]['content'],
            "language": language,  # Use detected language
            "sample_identifier": {
                "dataset_name": dataset_name,
                "hf_repo": hf_repo,
                "hf_index": hf_index,
                "hf_split": hf_split
            },
            "perplexity": perplexity,
            "classification_fields": {
                "question": task_data[question_key],
                "choices": choices,
                "ground_truth": {
                    "id": choices[task_data["answer"]]["id"],
                    "text": choices[task_data["answer"]]["text"]
                }
            },
            "prompt_logprobs": prompt_tokens_logprobs
        }

    def _get_guestion_index(self, df_map, row):
        """Find matching question and choices in the mapping DataFrame."""
        try:
            # Clean and normalize row choices
            row_choices = [answer.split(". ")[1] for answer in row['options']]
            row_choices_set = set(row_choices)  # Convert to set for comparison
            self.logger.debug(f"Row choices: {row_choices}")
            
            # First find questions that match
            question_mask = df_map['question'] == row['question']
            self.logger.debug(f"Question: {row['question']}")
            self.logger.debug(f"Number of matching questions: {question_mask.sum()}")
            choices_mask = df_map['choices'].apply(set).apply(lambda x: x == row_choices_set)
            matches = df_map[question_mask & choices_mask]
            self.logger.debug(f"Total matches found: {len(matches)}")

            if len(matches) == 0:
                self.logger.warning(f"No matches found for question: {row['question']}")
                self.logger.warning(f"Sample of df_map:\n{df_map.head()}")
                self.logger.warning("Attempting to find questions with different choices")

                # Log all questions with matching text
                question_matches = df_map[question_mask]
                self.logger.warning(f"Questions with same text found: {len(question_matches)}")

                # Log all choices for matching questions
                for i, df_row in question_matches.iterrows():
                    df_choices = df_row['choices']
                    if isinstance(df_choices, np.ndarray):
                        df_choices = df_choices.tolist()
                    elif isinstance(df_choices, str):
                        df_choices = df_choices.split(", ")
                    df_choices_set = set(df_choices)
                    self.logger.warning(f"Index {i} - choices in df: {sorted(df_choices_set)}")
                    self.logger.warning(f"Row choices: {sorted(row_choices_set)}")
                    self.logger.warning(f"Sets equal: {df_choices_set == row_choices_set}")
                    self.logger.warning(f"Difference: {df_choices_set.symmetric_difference(row_choices_set)}")

                self.logger.warning("No matching question-choices combination found, returning -1")
                return -1, 'test'  # Default to 'test' split when no match found

            match_data = matches.iloc[0]
            self.logger.debug(f"Match found - index: {match_data['index']}, source: {match_data['source']}")
            return match_data['index'], match_data['source']
        
        except Exception as e:
            self.logger.error(f"Error in _get_guestion_index: {str(e)}")
            self.logger.error(f"Row data: {row}")
            self.logger.error(f"DataFrame map shape: {df_map.shape}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _build_output_section(self, row: pd.Series, probs: bool) -> Dict:
        """Build output section of schema."""
        if probs:
            logprobs = row['logprobs']
            generated_tokens_logprobs = [
                self.add_token_index_to_tokens(logprob_dict['log_probs'])
                for logprob_dict in logprobs
            ]
        else:
            generated_tokens_logprobs = []
        return {
            "response": row['generated_text'],
            "cumulative_logprob": row['cumulative_logprob'],
            # "generated_tokens_ids": row['generated_text_tokens_ids'].tolist(),
            "generated_tokens_logprobs": generated_tokens_logprobs
        }

    def get_close_matches_from_row(self, task_data, row_data: pd.Series) -> str:
        choices = task_data['options']
        generated_answer = row_data['generated_text']
        ai_answer = generated_answer.strip().split("\n")
        ai_answer = ai_answer[0].strip()
        return get_close_matches(ai_answer, choices, n=1, cutoff=0.0)[0]

    def _build_evaluation_section(self, task_data, row: pd.Series) -> Dict:
        """Build evaluation section of schema."""

        return {
            "ground_truth": row['references'][0],
            "evaluation_method": {
                "method_name": "content_similarity",
                "description": "Finds the most similar answer among the given choices by comparing the textual content",
                "closest_answer": self.get_close_matches_from_row(task_data, row)
            },
            "score": row['scores']['score']
        }

    def convert_dataframe(self, df: pd.DataFrame, logger=None, probs=True) -> List[Dict]:
        conversion_map = {'prompt': json.loads, 'references': eval, 'scores': eval,
                          'run_generation_args': json.loads, 'run_model_args': json.loads}
        try:
            if logger:
                logger.debug(f"Starting DataFrame conversion with columns: {df.columns}")
            
            # Apply conversions in a vectorized manner
            for col, func in conversion_map.items():
                if col not in df.columns:
                    if logger:
                        logger.error(f"Missing required column: {col}")
                    raise KeyError(f"Missing required column: {col}")
                try:
                    df[col] = df[col].apply(func)
                except Exception as e:
                    if logger:
                        logger.error(f"Error converting column {col}: {str(e)}")
                        logger.error(f"Sample value: {df[col].iloc[0]}")
                    raise

            # Handle nested task_data conversion
            try:
                df['task_data'] = df['raw_input'].apply(lambda x: json.loads(json.loads(x)['task_data']))
            except Exception as e:
                if logger:
                    logger.error(f"Error converting task_data: {str(e)}")
                    logger.error(f"Sample raw_input: {df['raw_input'].iloc[0]}")
                raise

            # Drop raw_input column after extraction
            df = df.drop(columns=['raw_input'])

            # Convert rows in parallel using pandas apply
            return df.apply(lambda row: self.convert_row(row, probs, logger), axis=1).tolist()

        except Exception as e:
            if logger:
                logger.error(f"Error converting DataFrame: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise


def main():
    """Main execution function."""
    f = "data_2025-01-09T19_00_00+00_00_2025-01-09T23_00_00+00_00.parquet"
    parquet_path = Path(f"~/Downloads/{f}")
    models_metadata_path = Path(Constants.ExperimentConstants.MODELS_METADATA_PATH)

    # Initialize converter
    converter = SchemaConverter(models_metadata_path)

    # Process data in batches
    processor = RunOutputMerger(parquet_path, batch_size=1000)
    converted_data = []
    i = 0
    for batch in tqdm(processor.process_batches()):
        # Process single batch for testing
        batch = batch.head(1000)
        converted_data.extend(converter.convert_dataframe(batch))
        print(f"Batch {i} processed")
        i += 1
        if i == 10:
            break

    # Save results
    output_path = Path("data_sample_ibm.json")
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=4)

    print(f"Conversion complete: {len(converted_data)} samples processed")
    print(f"Results saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()
