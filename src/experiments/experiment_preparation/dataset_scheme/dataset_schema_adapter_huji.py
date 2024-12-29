import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from src.utils.Constants import Constants
from src.utils.Utils import Utils


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


def create_hash(text: str) -> str:
    """Create a hash of the input text."""
    return hashlib.sha256(text.encode()).hexdigest()


def get_model_metadata(model_name: str, models_metadata: Dict) -> Dict:
    """Get model metadata from the metadata file."""
    metadata = models_metadata.get(model_name, {})
    if not metadata:
        print(f"Warning: No metadata found for model {model_name}")
    return metadata


def create_sample_identifier(card_name: str, split: str, idx) -> Dict[str, Any]:
    """Create sample identifier from card name."""
    dataset_name = card_name.split('.')[1]  # e.g., 'mmlu' from 'cards.mmlu.abstract_algebra'
    return {
        "dataset_name": card_name.split('cards.')[1],
        "split": split,
        "hf_repo": f"cais/{dataset_name}",
        "hf_index": idx
    }


def create_choices_order_config(shuffle) -> Dict[str, Any]:
    """Create default choices order configuration."""
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
    return choices_order


def convert_to_schema_format(dataset: Dict[str, Any], max_new_tokens: int, data: Dict[str, Any],
                             template: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a single sample to the schema format."""
    with open(Constants.ExperimentConstants.MODELS_METADATA_PATH, 'r') as f:
        models_metadata = json.load(f)
    model_name = data["model_name"]
    model_metadata = get_model_metadata(model_name, models_metadata)

    samples = []
    for idx, result in enumerate(data["results"]["test"]):
        instance_text = result["Instance"]
        response = result["Result"]
        perplexity = None if "Perplexity" not in result else result["Perplexity"]
        sample = dataset['test'][result['Index']]

        question = eval(sample['task_data'])['question']
        options = eval(sample['task_data'])['options']
        choices = eval(sample['task_data'])['choices']
        choices = [(option.split(".")[0], choice) for option, choice in zip(options, choices)]

        # Create evaluation ID
        evaluation_id = create_hash(instance_text)

        # Create sample dictionary according to new schema
        formatted_sample = {
            "evaluation_id": evaluation_id,
            "model": {
                "model_info": {
                    "name": model_name,
                    "family": model_metadata.get("ModelInfo", {}).get("family")
                },
                "configuration": {
                    "architecture": model_metadata.get("Configuration", {}).get("architecture"),
                    "parameters": model_metadata.get("Configuration", {}).get("parameters"),
                    "context_window": model_metadata.get("Configuration", {}).get("context_window", 2048),
                    "is_instruct": model_metadata.get("Configuration", {}).get("is_instruct", False),
                    "hf_path": model_name,
                    "revision": model_metadata.get("Configuration", {}).get("revision")
                },
                "inference_settings": {
                    "quantization": {
                        "bit_precision": "int8",
                        "method": "dynamic"
                    },
                    "generation_args": {
                        "use_vllm": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": None,
                        "max_tokens": max_new_tokens,
                        "stop_sequences": []
                    }
                }
            },
            "prompt_config": {
                "prompt_class": "MultipleChoice",
                "format": {
                    "type": "MultipleChoice",
                    "template": template["input_format"],
                    "separator": SeparatorMap.get_separator(template["choices_separator"]),
                    "enumerator": template["enumerator"],
                    "choices_order": create_choices_order_config(template["shuffle_choices"]),
                    "shots": Utils.word_to_number("zero"),
                    "demos": None
                }
            },
            "instance": {
                "task_type": "classification",
                "raw_input": question,
                "sample_identifier": create_sample_identifier(data["card"], "test", idx),
                "token_ids": None,
                "perplexity": perplexity,
                "classification_fields": {
                    "full_input": instance_text,
                    "question": question,
                    "choices": [{"id": choice[0], "text": choice[1]} for choice in choices],
                    "ground_truth": {
                        "id": choices[eval(sample['task_data'])["answer"]][0],
                        "text": choices[eval(sample['task_data'])["answer"]][1]
                    }
                }
            },
            "output": {
                "response": response,
                "cumulative_logprob": None,
                "generated_tokens_ids": None,
                "generated_tokens_logprobs": [],
                "max_prob": response.split()[0]
            },
            "evaluation": {
                "ground_truth": result["GroundTruth"],
                "evaluation_method": {
                    "method_name": "content_similarity",
                    "description": "Finds the most similar answer among the given choices by comparing the textual content"
                },
                "score": result["Score"]
            }
        }
        samples.append(formatted_sample)
    return samples


def convert_dataset(dataset: Dict[str, Any], input_data: Dict[str, Any], template_data: Dict[str, Any]) -> List[
    Dict[str, Any]]:
    """Convert entire dataset to new format."""
    max_new_tokens = max([len(instance["target"].split()) for instance in dataset['test']]) * 2
    return convert_to_schema_format(dataset, max_new_tokens, input_data, template_data)


# Usage example:
if __name__ == "__main__":
    with open(
            '/Users/ehabba/PycharmProjects/LLM-Evaluation/results/MultipleChoiceTemplatesInstructions/Mistral-7B-Instruct-v0.2/mmlu.abstract_algebra/zero_shot/empty_system_format/experiment_template_0.json',
            'r') as f:
        input_data = json.load(f)
    with open(
            '/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/MultipleChoiceTemplatesStructured/mmlu.abstract_algebra/template_0.json',
            'r') as f:
        template_data = json.load(f)

    template_name = input_data['template_name']

    if template_name.startswith('template_'):
        template_name = "MultipleChoiceTemplatesInstructionsWithoutTopic/enumerator_capitals_choicesSeparator_comma_shuffleChoices_False"

    catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
    template_name.replace('MultipleChoiceTemplatesStructuredTopic',
                          'MultipleChoiceTemplatesStructuredWithTopic')
    template_name.replace('MultipleChoiceTemplatesStructured', 'MultipleChoiceTemplatesStructuredWithoutTopic')
    template_name = template_name.replace('MultipleChoiceTemplatesStructuredTopic',
                                          'MultipleChoiceTemplatesStructuredWithTopic')
    template_name = template_name.replace('MultipleChoiceTemplatesStructured',
                                          'MultipleChoiceTemplatesStructuredWithoutTopic')

    template_path = os.path.join(catalog_path, template_name + '.json')
    if not os.path.exists(template_path):
        template_path = Path(
            "/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/Catalog/MultipleChoiceTemplatesInstructionsWithoutTopic/enumerator_capitals_choicesSeparator_comma_shuffleChoices_False.json")

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

    converted_data = convert_dataset(dataset, input_data, template_data)

    with open('data_sample_huji.json', 'w') as f:
        json.dump(converted_data, f, indent=4)
