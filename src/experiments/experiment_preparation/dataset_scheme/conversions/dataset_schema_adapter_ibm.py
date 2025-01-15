import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.experiments.experiment_preparation.dataset_scheme.conversions.RunOutputMerger import RunOutputMerger
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
        return mapping[value]

    def calculate_perplexity(self, logprobs: List) -> float:
        """Calculate perplexity from log probabilities."""
        if logprobs[0] is None:
            logprobs = logprobs[1:]
        logprob_values = np.array([item[0]['logprob'] for item in logprobs])
        return np.exp(-np.mean(logprob_values))

    def convert_row(self, row: pd.Series, probs: bool = True, logger: Optional = None) -> Dict:
        """Convert a single DataFrame row to schema format."""
        task_data = row['task_data']
        recipe = self._parse_config_string(row['run_unitxt_recipe'])

        # Build schema sections
        schema = {
            "evaluation_id": row['id'],
            "model": self._build_model_section(
                row
            ),
            "prompt_config": self._build_prompt_section(task_data,
                                                        recipe,
                                                        logger=logger
                                                        ),
            "instance": self._build_instance_section(
                row, task_data, recipe, probs
            ),
            "output": self._build_output_section(row, probs),
            "evaluation": self._build_evaluation_section(row)
        }

        return schema

    def _parse_config_string(self, config_string: str) -> Dict:
        """Parse configuration string to dictionary."""
        return dict(pair.split('=') for pair in config_string.split(','))

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
        return template_data['input_format']

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

    def _build_prompt_section(self, task_data: Dict, recipe: Dict,
                              logger=None) -> Dict:
        """Build prompt configuration section of schema."""
        # Process demonstration examples
        enumerator, separator, choices_order = self.parse_template_params(recipe)
        demos = ([] if 'demos' not in task_data else
                 self.transform_demos(task_data['demos']))
        return {
            "prompt_class": "MultipleChoice",
            "format": {
                "template": self._get_template(recipe, logger),
                "separator": separator,
                "enumerator": enumerator,
                "choices_order": choices_order,
                "shots": int(recipe['num_demos']),
                "demos": demos
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

            prompt_tokens_logprobs= [
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
        map_file_path = Path("hf_map_data") / f"{map_file_name}_samples.json"
        hf_repo = f"cais/{recipe['card'].split('.')[1]}" if map_file_name == "mmlu" else f"Rowan/hellaswag" if map_file_name == "hellaswag" else f"allenai/{map_file_name}"
        if map_file_name not in self._index_map_cache:
            with open(map_file_path, 'r') as file:
                index_map = json.load(file)
                self._index_map_cache[map_file_name] = index_map
        else:
            index_map = self._index_map_cache[map_file_name]

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
        else:
            hf_repo = None
        question_key = 'question' if 'question' in task_data else 'context'
        return {
            "task_type": "classification",
            "raw_input": row['prompt'],
            "language": "en",
            "sample_identifier": {
                "dataset_name": recipe['card'].split("cards.")[1],
                "hf_repo": hf_repo,
                "hf_index": index_map[task_data[question_key]]['index'],
                "hf_split": index_map[task_data[question_key]]['source']
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
            "generated_tokens_ids": row['generated_text_tokens_ids'].tolist(),
            "generated_tokens_logprobs": generated_tokens_logprobs
        }

    def _build_evaluation_section(self, row: pd.Series) -> Dict:
        """Build evaluation section of schema."""
        return {
            "ground_truth": row['references'][0],
            "evaluation_method": {
                "method_name": "content_similarity",
                "description": "Finds the most similar answer among the given choices by comparing the textual content"
            },
            "score": row['scores']['score']
        }

    def convert_dataframe(self, df: pd.DataFrame, logger=None, probs=True) -> List[Dict]:
        conversion_map = {'prompt': eval, 'references': eval, 'scores': eval, 'run_generation_args': json.loads,
                          'run_model_args': json.loads, }

        try:
            # Apply conversions in a vectorized manner
            for col, func in conversion_map.items():
                df[col] = df[col].apply(func)

            # Handle nested task_data conversion
            df['task_data'] = df['raw_input'].apply(lambda x: eval(eval(x)['task_data']))

            # Drop raw_input column after extraction
            df = df.drop(columns=['raw_input'])

            # Convert rows in parallel using pandas apply
            return df.apply(lambda row: self.convert_row(row, probs, logger), axis=1).tolist()

        except Exception as e:
            if logger:
                logger.error(f"Error converting DataFrame: {str(e)}")
            raise


def main():
    """Main execution function."""
    f = "data_2025-01.parquet"
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
