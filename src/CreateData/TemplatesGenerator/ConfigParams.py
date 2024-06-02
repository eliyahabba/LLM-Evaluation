import random

from src.utils.DatasetsManger import DatasetsManger

random.seed(42)
from src.utils.MMLUData import MMLUData


def get_mmlu_instructions(topic: str) -> str:
    return f"The following are multiple choice questions (with answers) about {topic}.\n"


def get_structured_mmlu_instructions() -> str:
    return f"Question: [question] Choices: [choices] Answer: [answer]\n"


class ConfigParams:
    base_args_sciq = {
        "input_format": "Context: [context] Question: [question] Choices: [choices] Answer: [answer]\nContext: {context} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_separator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}. {choice_text}",
        "shuffle_choices": False,
        "postprocessors": [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
    }

    base_args_race = {
        "input_format": "Context: [context] Question: [question] Choices: [choices] Answer: [answer]\nContext: {context} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_separator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}. {choice_text}",
        "shuffle_choices": False,
        "postprocessors": [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
    }

    base_args_ai2_arc_easy = {
        "input_format": "Topic: [topic] Question: [question] Choices: [choices] Answer: [answer]\nTopic: {topic} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_separator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}. {choice_text}",
        "shuffle_choices": False,
        "postprocessors": [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
    }

    base_args_mmlu_structured = {
        "input_format": get_structured_mmlu_instructions() + "Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_separator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}. {choice_text}",
        "shuffle_choices": False,
        "postprocessors": [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
    }

    base_args_mmlu_instructions = {
        "input_format": "The following are multiple choice questions (with answers) about {topic}.\n"
                        + "Question: {question} Answers: {choices}\nAnswer:\n",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_separator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}. {choice_text}",
        "shuffle_choices": False,
        "postprocessors": [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
    }

    base_args_hellaswag = {
        "input_format": "Context: [context] Question: [question] Choices: [choices] Answer: [answer]\nContext: {context} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_separator": "\n",
        "enumerator": "capitals",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}. {choice_text}",
        "shuffle_choices": False,
        "postprocessors": [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
    }

    datasets_templates = [base_args_sciq, base_args_race, base_args_ai2_arc_easy, base_args_hellaswag]
    dataset_names_to_templates = dict(zip(DatasetsManger.get_base_dataset_names(), datasets_templates))
    for mmlu_dataset in MMLUData.get_mmlu_datasets():
        dataset_names_to_templates[f"{mmlu_dataset}"] = base_args_mmlu_instructions

    override_options = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman"],
        "choices_separator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
        "shuffle_choices": [False, True],
        # Add more parameters and their possible values as needed
    }
    map_enumerator = {"ABCDEFGHIJKLMNOP": "capitals",
                      "abcdefghijklmnop": "lowercase",
                      str(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                           '18', '19', '20']): "numbers",
                      str(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
                           'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']): "roman"}
