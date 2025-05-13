import itertools

import json
from typing import Dict, List, Any


class PromptOptions:
    GREEK_CHARS = "αβγδεζηθικ"  # 10 Greek letters
    KEYBOARD_CHARS = "!@#$%^₪*)("  # 26 lowercase letters
    SHUFFLE_CHOICES_COMBINATIONS = {
        "False": {"shuffle_choices": False, "sort_choices_by_length": False, "sort_choices_alphabetically": False,
                  "reverse_choices": False, "place_correct_choice_position": None},
        "lengthSort": {"shuffle_choices": False, "sort_choices_by_length": True,
                       "sort_choices_alphabetically": False,
                       "reverse_choices": False, "place_correct_choice_position": None},
        "lengthSortReverse": {"shuffle_choices": False, "sort_choices_by_length": True,
                              "sort_choices_alphabetically": False, "reverse_choices": True,
                              "place_correct_choice_position": None},
        "alphabeticalSort": {"shuffle_choices": False, "sort_choices_alphabetically": True,
                             "sort_choices_by_length": False, "reverse_choices": False,
                             "place_correct_choice_position": None},
        "alphabeticalSortReverse": {"shuffle_choices": False, "sort_choices_alphabetically": True,
                                    "sort_choices_by_length": False, "reverse_choices": True,
                                    "place_correct_choice_position": None},
        "placeCorrectChoiceFirst": {"shuffle_choices": False, "sort_choices_alphabetically": False,
                                    "sort_choices_by_length": False, "reverse_choices": False,
                                    "place_correct_choice_position": 0},
        "placeCorrectChoiceFourth": {"shuffle_choices": False, "sort_choices_alphabetically": False,
                                      "sort_choices_by_length": False, "reverse_choices": False,
                                      "place_correct_choice_position": -1},
    }

    override_options = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman", KEYBOARD_CHARS, GREEK_CHARS],

        "choices_separator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
        "shuffle_choices": list(SHUFFLE_CHOICES_COMBINATIONS.keys()),
        # Add more parameters and their possible values as needed
    }

    ENUM_CHARS = {"ABCDEFGHIJKLMNOP": "capitals",
                  "abcdefghijklmnop": "lowercase",
                  str(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19', '20']): "numbers",
                  str(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
                       'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']): "roman",
                  KEYBOARD_CHARS: "keyboard",  # Added mapping for keyboard chars
                  GREEK_CHARS: "greek"  # Added mapping for greek chars
                  }

    @classmethod
    def get_options(cls) -> Dict[str, List]:
        return cls.override_options

    @classmethod  # Changed to classmethod to access class attributes
    def format_value(cls, value: Any) -> str:
        """Format a value for use in template name"""
        if isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            # Special characters mapping
            separator_names = {
                " ": "space",
                "\n": "newline",
                " | ": "pipe",
                " OR ": "OrCapital",
                " or ": "orLower",
                ", ": "comma",
                "; ": "semicolon"
            }

            # Check separators first
            if value in separator_names:
                return separator_names[value]

            # Check special character sets
            if value in cls.ENUM_CHARS:
                return cls.ENUM_CHARS[value]

            # For any other string values
            return value.replace(" ", "")

        return str(value)

    @classmethod
    def generate_template_name(cls, combination: Dict[str, Any]) -> str:
        """Generate a descriptive template name from a combination of options"""
        parts = []
        for key, value in combination.items():
            formatted_value = cls.format_value(value)
            camel_key = cls.to_camel_case(key)
            parts.append(f"{camel_key}_{formatted_value}")
        return "_".join(parts)

    @staticmethod
    def to_camel_case(snake_str: str) -> str:
        """Convert snake_case to camelCase"""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    @classmethod
    def generate_template_combinations(cls, dataset_name) -> Dict[str, Dict]:
        """Generate all possible combinations of template options"""
        options = cls.get_options()
        all_combinations = [
            {key: value for key, value in zip(options.keys(), values)}
            for values in itertools.product(*options.values())
        ]
        if dataset_name in ["global_mmlu"]:
            global_mmlu_options = cls.get_options()
            global_mmlu_options["enumerator"] = ['capitals', 'numbers', 'roman', '!@#$%^₪*)(', 'αβγδεζηθικ']
            global_mmlu_options["choices_separator"] = [' ', '\n', ', ', '; ', ' OR ']

            global_mmlu_combinations = [
                {key: value for key, value in zip(global_mmlu_options.keys(), values)}
                for values in itertools.product(*global_mmlu_options.values())
            ]
            return {
                cls.generate_template_name(combo): combo
                for combo in global_mmlu_combinations
            }


        if dataset_name in ["mmlu", "ai2_arc.arc_easy", "ai2_arc.arc_challenge", "hellaswag", "openbook_qa",
                            "race_high", "race_middle", "quailty"]:
            return {
                cls.generate_template_name(combo): combo
                for combo in all_combinations
            }
        elif dataset_name in ["mmlu_pro", "social_iqa"]:
            combinations = [combo for combo in all_combinations if
                            combo["shuffle_choices"] != "placeCorrectChoiceFourth"]
            return {
                cls.generate_template_name(combo): combo
                for combo in combinations
            }
        else:
            raise ValueError(f"Dataset name {dataset_name} is not supported")


def get_prompt_paraphrasing(dataset_name):
    if dataset_name in ["mmlu", "mmlu_pro",  "global_mmlu", "ai2_arc.arc_easy", "ai2_arc.arc_challenge", "openbook_qa",
                        "social_iqa"]:
        return ['MultipleChoiceTemplatesInstructionsWithTopic', 'MultipleChoiceTemplatesInstructionsWithoutTopicFixed',
                'MultipleChoiceTemplatesInstructionsWithTopicHelm',
                'MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed',
                'MultipleChoiceTemplatesInstructionsWithoutTopicHarness', 'MultipleChoiceTemplatesStructuredWithTopic',
                'MultipleChoiceTemplatesStructuredWithoutTopic', 'MultipleChoiceTemplatesInstructionsProSASimple',
                'MultipleChoiceTemplatesInstructionsProSAAddress', 'MultipleChoiceTemplatesInstructionsProSACould',
                'MultipleChoiceTemplatesInstructionsStateHere', 'MultipleChoiceTemplatesInstructionsStateBelow',
                'MultipleChoiceTemplatesInstructionsStateBelowPlease']
    if dataset_name in ["hellaswag"]:
        return \
            ['MultipleChoiceTemplatesInstructionsStandard', 'MultipleChoiceTemplatesInstructionsContext',
             'MultipleChoiceTemplatesInstructionsStructured', 'MultipleChoiceTemplatesInstructionsBasic',
             'MultipleChoiceTemplatesInstructionsState1', 'MultipleChoiceTemplatesInstructionsState2',
             'MultipleChoiceTemplatesInstructionsState3', 'MultipleChoiceTemplatesInstructionsState4',
             'MultipleChoiceTemplatesInstructionsState5', 'MultipleChoiceTemplatesInstructionsState6',
             'MultipleChoiceTemplatesInstructionsState7', 'MultipleChoiceTemplatesInstructionsState8']
    if dataset_name in ["race_high", "race_middle", "quailty"]:
        return ["MultipleChoiceContextTemplateBasic",
                "MultipleChoiceContextTemplateBasicNoContextLabel",
                "MultipleChoiceContextTemplateMMluStyle",
                "MultipleChoiceContextTemplateMMluHelmStyle",
                "MultipleChoiceContextTemplateMMluHelmWithChoices",
                "MultipleChoiceContextTemplateProSASimple",
                "MultipleChoiceContextTemplateProSACould",
                "MultipleChoiceContextTemplateStateNumbered",
                "MultipleChoiceContextTemplateStateOptions",
                "MultipleChoiceContextTemplateStateSelect",
                "MultipleChoiceContextTemplateStateRead",
                "MultipleChoiceContextTemplateStateMultipleChoice"
                ]


def create_experiments_json():
    datasets = ["mmlu", "mmlu_pro",  "global_mmlu",
                "ai2_arc.arc_easy", "ai2_arc.arc_challenge", "hellaswag", "openbook_qa", "social_iqa",
                "race_high","race_middle", "quailty"]
    catalog_dataset_map = {
        "mmlu": "MMLU",
        "mmlu_pro": "MMLU_PRO",
        "global_mmlu": "MMLU",
        "ai2_arc.arc_easy": "AI2_ARC",
        "ai2_arc.arc_challenge": "AI2_ARC",
        "hellaswag": "HellaSwag",
        "openbook_qa": "OpenBookQA",
        "social_iqa": "Social_IQa",
        "race_high": "Race",
        "race_middle": "Race",
        "quailty": "QuALITY"
    }

    experiments_config = {}

    for dataset_name in datasets:
        prompt_paraphrases = get_prompt_paraphrasing(dataset_name)
        templates = PromptOptions.generate_template_combinations(
            dataset_name)
        template_names = list(templates.keys())
        if dataset_name not in ["mmlu", "global_mmlu", "hellaswag","openbook_qa", "race_high", "race_middle", "quailty"]:
            # remove all the the templates that have the shuffle_choices as placeCorrectChoiceFourth ot placeCorrectChoiceFirst
            template_names = [template_name for template_name in template_names if ("placeCorrectChoiceFourth" not
                                                                                        in template_name)]
        experiments_config[dataset_name] = {
            "dataset_catalog_name": catalog_dataset_map[dataset_name],
            "prompt_paraphrases": prompt_paraphrases,
            "template_names": template_names
        }

    with open('experiments_config.json', 'w', encoding='utf-8') as f:
        json.dump(experiments_config, f, indent=4)

    return experiments_config




if __name__ == "__main__":
    # run_experiments_from_config()
    create_experiments_json()
