import json
from typing import Dict, List, Any
import re


def create_sample_identifier(card_name: str, split: str) -> Dict[str, Any]:
    """Create sample identifier from card name."""
    dataset_name = card_name.split('.')[1]  # e.g., 'mmlu' from 'cards.mmlu.abstract_algebra'
    return {
        "dataset_name": dataset_name,
        "split": split,
        "hf_repo": f"lukaemon/{dataset_name}",  # assumption
        "hf_index": 0  # will be updated in the conversion loop
    }


def convert_to_schema_format(data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the schema format."""

    # Helper to generate fake token IDs (for demonstration)
    def generate_token_ids(length: int) -> List[int]:
        return list(range(1000, 1000 + length))

    # Helper to create token logprobs
    def create_token_logprobs(response: str) -> List[Dict[str, Any]]:
        tokens = response.split()[:3]  # Take first 3 tokens for example
        result = []
        for token in tokens:
            logprob_entry = {
                "token": token,
                "token_id": generate_token_ids(1)[0],
                "top_5": []
            }
            # Create top 5 predictions
            for rank in range(1, 6):
                logprob_entry["top_5"].append({
                    "token": f"{token}_{rank}",
                    "token_id": generate_token_ids(1)[0],
                    "logprob": -0.1 * rank,
                    "rank": rank
                })
            result.append(logprob_entry)
        return result

    # Process each result
    for idx, result in enumerate(data["results"]["test"]):
        instance_text = result["Instance"]
        response = result["Result"]
        perplexity = None if "Perplexity" not in result else result["Perplexity"]
        # Extract question and choices from the instance text
        match = re.search(r"Question: (.*?)Answers: (.*?)Answer:", instance_text, re.DOTALL)
        if not match:
            continue

        question = match.group(1).strip()
        choices = match.group(2).strip()
        choices = [('A', '0'), ('B','4'), ('C', '2'), ('D', '6')]  # Example format
        return {
            "EvaluationId": str(idx),
            "Model": {
                "Name": data["model_name"],
                "Quantization": {
                    "QuantizationBit": "8bit",
                    "QuantizationMethod": "dynamic"
                },
                "GenerationArgs": {
                    "max_tokens": 10,
                    "do_sample": False,
                    "top_p": 1,
                    "temperature": 0.7
                }

            },
            "Prompt": {
                "Instruct Template": template["input_format"],
                "Separator": template["choices_separator"],
                "Enumerator": template["enumerator"],
                "ShuffleChoices": template["shuffle_choices"],
                "Shots": "zero",
                "TokenIds": None
                    # generate_token_ids(10)
            },
            "Instance": {
                "TaskType" : "classification",
                "RawInput": instance_text,
                "SampleIdentifier": create_sample_identifier(data["card"], "test"),
                "Perplexity": perplexity,
                "ClassificationFields": {
                "Topic": data["card"].split('.')[-1],  # e.g., 'abstract_algebra'
                "Question": question,
                "Choices": [{
                    "id": choice[0],
                    "text": choice[1]
                } for choice in choices],  # Example format
                "GroundTruth": result["GroundTruth"]
                }
            },
            "Output": {
                "Response": response,
                "CumulativeLogprob": None,
                    # -2.34,  # Example value
                "GeneratedTokenIds": None ,
                    # generate_token_ids(len(response.split())),
                "TokenLogprobs": None ,
                # create_token_logprobs(response),
                "MaxProb": response.split()[0]  # Taking first token as max prob answer
            },
            "Evaluation": {
                "GroundTruth": result["GroundTruth"],
                "Score": result["Score"]
            }
        }


def convert_dataset(input_data: Dict[str, Any], template_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert entire dataset to new format."""
    converted_samples = []
    for idx, result in enumerate(input_data["results"]["test"]):
        sample = convert_to_schema_format(input_data, template_data)
        if sample:
            sample["Instance"]["SampleIdentifier"]["hf_index"] = idx
            converted_samples.append(sample)

    return converted_samples


# Usage example:
if __name__ == "__main__":
    with open('/Users/ehabba/PycharmProjects/LLM-Evaluation/results/MultipleChoiceTemplatesInstructions/Mistral-7B-Instruct-v0.2/mmlu.abstract_algebra/zero_shot/empty_system_format/experiment_template_0.json', 'r') as f:
        input_data = json.load(f)
    with open('/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/MultipleChoiceTemplatesStructured/mmlu.abstract_algebra/template_0.json', 'r') as f:
        template_data = json.load(f)

    converted_data = convert_dataset(input_data, template_data)

    with open('converted_data.json', 'w') as f:
        json.dump(converted_data, f, indent=2)