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

    def _load_models_metadata(self) -> None:
        """Load models metadata from file."""
        with open(self.models_metadata_path) as f:
            self.models_metadata = json.load(f)

    def transform_logprobs(self, logprobs: Union[str, List]) -> Optional[List]:
        """Transform log probabilities to schema format."""
        transformed = []
        for token_dict in logprobs:
            if token_dict == 'None':
                transformed.append(None)
                continue
            transformed.append(self.add_token_index_to_tokens(token_dict))
        return transformed

    def add_token_index_to_tokens(self, token_dict: Dict) -> List[Dict]:
        """Transform log probabilities to schema format."""
        transformed_token_data = token_dict['log_probs'].tolist()
        return transformed_token_data

    def transform_demos(self, demos: List[Dict]) -> List[Dict]:
        """Transform demonstration examples to schema format."""
        transformed = []
        for demo in demos:
            choices = [
                {"id": opt.split(".")[0], "text": choice}
                for opt, choice in zip(demo['options'], demo['choices'])
            ]

            transformed_demo = {
                "topic": None if 'topic' not in demo else demo['topic'],
                "question": demo["question"],
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
        shuffle = config.split('shuffleChoices_')[1] == 'True'
        if shuffle:
            choices_order = {
                "method": "random",
                "description": "Randomly shuffles all choices"
            }
        else:
            choices_order = {
                "method": "none",
                "description": "Preserves the original order of choices as provided"
            }

        separator = SeparatorMap.get_separator(choice_sep)
        return enumerator, separator, choices_order

    def parse_generation_args(self, args_str: str) -> Dict:
        """Parse generation arguments from string format."""
        args = json.loads(args_str)
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
        raw_input = json.loads(row['raw_input'])
        task_data = eval(raw_input['task_data'])
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
        """Extract template format from run_unitxt_recipe."""
        template_name = run_unitxt_recipe['template'].split('huji_workshop.')[1].split(".")[0]
        template_name = run_unitxt_recipe['template'].split('templates.huji_workshop.')[-1]
        logger.info(f"Template name: {template_name}")
        catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
        logger.info(f"Catalog path: {catalog_path}")
        template_name = template_name.replace(".", "/")
        logger.info(f"Template name: {template_name}")

        template_path = Path(catalog_path, template_name + '.json')
        logger.info(f"Template path: {template_path}")
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        return template_data['input_format']

    def _build_model_section(self, row: pd.Series
                             ) -> Dict:
        """Build model section of schema."""

        # Get model configuration
        model_args = json.loads(row['run_model_args'])
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
                "template": self._get_template(recipe,logger),
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
            prompt_tokens_logprobs = [
                self.add_token_index_to_tokens(prompt_logprobs[i])
                for i in range(len(prompt_logprobs))
            ]
            perplexity = self.calculate_perplexity(prompt_tokens_logprobs)
        else:
            prompt_tokens_logprobs = []
            perplexity = None

        with open(f"{recipe['card'].split('.')[1]}_samples.json", 'r') as file:
            index_map = json.load(file)
        return {
            "task_type": "classification",
            "raw_input": eval(row['prompt']),
            "language": "en",
            "sample_identifier": {
                "dataset_name": recipe['card'].split("cards.")[1],
                "split": "test",
                "hf_repo": f"cais/{recipe['card'].split('.')[1]}",
                "hf_index": index_map[task_data['question']]
            },
            "perplexity": perplexity,
            "classification_fields": {
                "question": task_data['question'],
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
                self.add_token_index_to_tokens(logprobs[i])
                for i in range(len(logprobs))
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
            "ground_truth": eval(row['references'])[0],
            "evaluation_method": {
                "method_name": "content_similarity",
                "description": "Finds the most similar answer among the given choices by comparing the textual content"
            },
            "score": eval(row['scores'])['score']
        }

    def convert_dataframe(self, df: pd.DataFrame, logger=None, probs=True) -> List[Dict]:
        """Convert entire DataFrame to schema format."""
        return [self.convert_row(row, probs,logger) for _, row in df.iterrows()]


def main():
    """Main execution function."""
    f = "data_2025-01.parquet"
    parquet_path = Path(f"~/Downloads/{f}")
    models_metadata_path = Path(Constants.ExperimentConstants.MODELS_METADATA_PATH)

    # Initialize converter
    converter = SchemaConverter(models_metadata_path)

    # Process data in batches
    processor = RunOutputMerger(parquet_path)
    converted_data = []

    for batch in tqdm(processor.process_batches()):
        # Process single batch for testing
        batch = batch.head(10)
        converted_data.extend(converter.convert_dataframe(batch))
        break

    # Save results
    output_path = Path("data_sample_ibm.json")
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=4)

    print(f"Conversion complete: {len(converted_data)} samples processed")
    print(f"Results saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()
