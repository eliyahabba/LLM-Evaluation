import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from src.utils.Utils import Utils


def create_hash(text: str) -> str:
    """Create a hash of the input text."""
    return hashlib.sha256(text.encode()).hexdigest()


def create_sample_identifier(card_name: str, split: str, idx) -> Dict[str, Any]:
    """Create sample identifier from card name."""
    dataset_name = card_name.split('.')[1]  # e.g., 'mmlu' from 'cards.mmlu.abstract_algebra'
    return {
        "dataset_name": dataset_name,
        "split": split,
        "hf_repo": f"cais/{dataset_name}",  # assumption
        "hf_index": idx  # will be updated in the conversion loop
    }


def get_model_metadata(model_name: str, models_metadata: Dict) -> Dict:
    """Get model metadata from the metadata file."""
    metadata = models_metadata.get(model_name, {})
    if not metadata:
        print(f"Warning: No metadata found for model {model_name}")
    return metadata


def convert_to_schema_format(dataset, max_new_tokens, data: Dict[str, Any], template: Dict[str, Any]) -> List[
    Dict[str, Any]]:
    """Convert a single sample to the schema format."""

    # Helper to generate fake token IDs (for demonstration)
    def generate_token_ids(length: int) -> List[int]:
        return list(range(1000, 1000 + length))

    # Helper to create token LogProbs
    def create_token_LogProbs(response: str) -> List[Dict[str, Any]]:
        tokens = response.split()[:3]  # Take first 3 tokens for example
        result = []
        for token in tokens:
            LogProb_entry = {
                "token": token,
                "token_id": generate_token_ids(1)[0],
                "top_5": []
            }
            # Create top 5 predictions
            for rank in range(1, 6):
                LogProb_entry["top_5"].append({
                    "token": f"{token}_{rank}",
                    "token_id": generate_token_ids(1)[0],
                    "LogProb": -0.1 * rank,
                    "rank": rank
                })
            result.append(LogProb_entry)
        return result

    # Process each result
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
        # Extract question and choices from the instance text
        # match = re.search(r"Question: (.*?)Answers: (.*?)Answer:", instance_text, re.DOTALL)
        # if not match:
        #     continue

        question = eval(sample['task_data'])['question']
        options = eval(sample['task_data'])['options']
        choices = eval(sample['task_data'])['choices']
        choices = [(option.split(".")[0], choice) for option, choice in zip(options, choices)]
        samples.append({
            "EvaluationId": create_hash(instance_text),
            "Model": {
                "ModelInfo": model_metadata.get("ModelInfo", {
                    "name": model_name,
                    "family": None
                }),
                "Configuration": model_metadata.get("Configuration", {
                    "architecture": None,
                    "parameters": None,
                    "context_window": None,
                    "is_instruct": None,
                    "hf_path": model_name,
                    "revision": None
                }),
                "InferenceSettings": {
                    "quantization": {
                        "bit_precision": "8bit",
                        "method": "dynamic"
                    },
                    "generation_args": {
                        "max_tokens": max_new_tokens,
                        "temperature": 0,
                        "top_p": 1,
                        "use_vllm": True  # או מה שמתאים לפי המערכת שלך
                    }
                }
            },
            "Prompt": {
                "promptClass": "MultipleChoice",
                "format": {
                    "type": "MultipleChoice",
                    "template": template["input_format"],
                    "separator": template["choices_separator"],
                    "enumerator": template["enumerator"],
                    "shuffleChoices": template["shuffle_choices"],
                    "shots": Utils.word_to_number("zero")
                }
            },
            "Instance": {
                "TaskType": "classification",
                "RawInput": question,
                "SampleIdentifier": create_sample_identifier(data["card"], "test", idx),
                "TokenIds": None,  # אולי כדאי להוסיף פונקציה שמייצרת
                "Perplexity": perplexity,
                "ClassificationFields": {
                    "Topic": data["card"].split('.')[-1],
                    "FullInput": instance_text,
                    "Question": question,
                    "Choices": [{
                        "id": choice[0],
                        "text": choice[1]
                    } for choice in choices],
                    "GroundTruth": {
                        "id": result["GroundTruth"].split(".")[0],
                        "text": choices[eval(sample['task_data'])["answer"]][1]
                    }
                }
            },
            "Output": {
                "Response": response,
                "CumulativeLogProb": None,
                "GeneratedTokensIds": None,
                "GeneratedTokensLogProbs": None,
                "MaxProb": response.split()[0]
            },
            "Evaluation": {
                "EvaluationMethod": {
                    "method_name": "content_similarity",
                    "description": "Finds the most similar answer among the given choices by comparing the textual content"
                },
                "GroundTruth": result["GroundTruth"],
                "Score": result["Score"]
            }
        })
    return samples


def convert_dataset(dataset, input_data: Dict[str, Any], template_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert entire dataset to new format."""
    # convert_dataseted_samples = []
    max_new_tokens = max([len(instance["target"].split()) for instance in dataset['test']]) * 2
    samples = convert_to_schema_format(dataset, max_new_tokens, input_data, template_data)
    return samples


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
    from src.utils.Constants import Constants

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

    with open('converted_data.json', 'w') as f:
        json.dump(converted_data, f, indent=2)
