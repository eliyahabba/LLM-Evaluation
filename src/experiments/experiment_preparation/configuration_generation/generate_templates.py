import argparse
import itertools
import os
from enum import Enum
from typing import Dict, List, Any, Optional


class BaseConfig:
    """Parent configuration class containing all config-related classes and methods"""

    class DatasetNames:
        """Dataset name constants"""
        MMLU = "mmlu"
        MMLU_PRO = "mmlu_pro"
        AI2_ARC_EASY = "ai2_arc.arc_easy"
        AI2_ARC_CHALLENGE = "ai2_arc.arc_challenge"
        HellaSwag = "hellaswag"
        OpenBookQA = "openbook_qa"
        Social_IQa = "social_iqa"

        @classmethod
        def get_datasets(cls) -> List[str]:
            return [cls.MMLU, cls.MMLU_PRO, cls.AI2_ARC_EASY, cls.AI2_ARC_CHALLENGE, cls.HellaSwag, cls.OpenBookQA,
                    cls.Social_IQa]

    class DatasetSubsets:
        """Dataset subset configurations"""

        @classmethod
        def get_subsets(cls) -> Dict[str, List[str]]:
            return {
                BaseConfig.DatasetNames.MMLU_PRO: [
                    "history", "law", "health", "physics", "business", "other",
                    "philosophy", "psychology", "economics", "math", "biology",
                    "chemistry", "computer_science", "engineering",
                ],
                BaseConfig.DatasetNames.MMLU: [
                    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
                    "clinical_knowledge", "college_biology", "college_chemistry",
                    "college_computer_science", "college_mathematics", "college_medicine",
                    "college_physics", "computer_security", "conceptual_physics",
                    "econometrics", "electrical_engineering", "elementary_mathematics",
                    "formal_logic", "global_facts", "high_school_biology",
                    "high_school_chemistry", "high_school_computer_science",
                    "high_school_european_history", "high_school_geography",
                    "high_school_government_and_politics", "high_school_macroeconomics",
                    "high_school_mathematics", "high_school_microeconomics",
                    "high_school_physics", "high_school_psychology",
                    "high_school_statistics", "high_school_us_history",
                    "high_school_world_history", "human_aging", "human_sexuality",
                    "international_law", "jurisprudence", "logical_fallacies",
                    "machine_learning", "management", "marketing", "medical_genetics",
                    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
                    "philosophy", "prehistory", "professional_accounting",
                    "professional_law", "professional_medicine", "professional_psychology",
                    "public_relations", "security_studies", "sociology",
                    "us_foreign_policy", "virology", "world_religions"
                ]
            }

    class PromptParaphrases:
        """Prompt template constants"""
        MC_WITH_TOPIC = "MultipleChoiceTemplatesInstructionsWithTopic"
        MC_WITHOUT_TOPIC = "MultipleChoiceTemplatesInstructionsWithoutTopicFixed"
        MC_WITH_TOPIC_HELM = "MultipleChoiceTemplatesInstructionsWithTopicHelm"
        MC_WITHOUT_TOPIC_HELM = "MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed"

        @classmethod
        def get_all_prompt_paraphrases(cls) -> List[str]:
            return [
                cls.MC_WITH_TOPIC,
                cls.MC_WITHOUT_TOPIC,
                cls.MC_WITH_TOPIC_HELM,
                cls.MC_WITHOUT_TOPIC_HELM
            ]

    class FewShot(Enum):
        """Few-shot learning parameters"""
        ZERO_SHOT = 0
        FEW_SHOT = 5

        @classmethod
        def get_values(cls) -> List[int]:
            return [shot.value for shot in cls]

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
                                         "place_correct_choice_position": 4},
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
            if dataset_name in ["mmlu", "ai2_arc.arc_easy", "ai2_arc.arc_challenge", "hellaswag", "openbook_qa"]:
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

    class ModelConfigs:
        """Model configurations"""
        PRIORITY_ORDER = ["Llama", "Mistral", "OLMo", "Gemma", "Qwen"]

        @classmethod
        def get_configs(cls) -> Dict[str, Dict[str, Dict[str, str]]]:
            return {
                "Llama": {
                    "Llama-3.2-1B-Instruct": {"path": "meta-llama/Llama-3.2-1B-Instruct", "size": "1B"},
                    "Llama-3.2-3B-Instruct": {"path": "meta-llama/Llama-3.2-3B-Instruct", "size": "3B"},
                    "Meta-Llama-3-8B-Instruct": {"path": "meta-llama/Meta-Llama-3-8B-Instruct", "size": "8B"},
                    "Meta-Llama-3-70B-Instruct": {"path": "meta-llama/Meta-Llama-3-70B-Instruct", "size": "70B"}
                },
                "Mistral": {
                    "Mistral-7B-Instruct-v0.3": {"path": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B"},
                    "Mixtral-8x7B-Instruct-v0.1": {"path": "mistralai/Mixtral-8x7B-Instruct-v0.1", "size": "8x7B"},
                    "Mixtral-8x22B-Instruct-v0.1": {"path": "mistralai/Mixtral-8x22B-Instruct-v0.1", "size": "8x22B"}
                },
                "OLMo": {
                    "OLMo-7B-Instruct": {"path": "allenai/OLMo-7B-Instruct", "size": "7B"},
                    "OLMoE-1B-7B-0924-Instruct": {"path": "allenai/OLMoE-1B-7B-0924-Instruct", "size": "1B"}
                },
                "Qwen": {
                    "Qwen2.5-0.5B-Instruct": {"path": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B"},
                    "Qwen2.5-72B-Instruct": {"path": "Qwen/Qwen2.5-72B-Instruct", "size": "72B"}
                }
            }

        @classmethod
        def get_priority_paths(cls) -> List[str]:
            configs = cls.get_configs()
            return [
                model_info["path"]
                for family in cls.PRIORITY_ORDER
                if family in configs
                for model_info in configs[family].values()
            ]

        @classmethod
        def get_model_family(cls, model_path: str) -> Optional[str]:
            """
            Find the model family for a given model path.

            Args:
                model_path: The full path of the model (e.g., 'meta-llama/Llama-3.2-1B-Instruct')

            Returns:
                str: The model family name if found, None otherwise
            """
            configs = cls.get_configs()

            # Iterate through each family and its models
            for family, models in configs.items():
                # Check if the model path exists in any of the family's models
                if any(model_info["path"] == model_path for model_info in models.values()):
                    return family

            return None


def get_prompt_paraphrasing(dataset_name):
    if dataset_name in ["mmlu", "mmlu_pro", "ai2_arc.arc_easy", "ai2_arc.arc_challenge", "hellaswag", "openbook_qa",
                        "social_iqa"]:
        return ['MultipleChoiceTemplatesInstructionsWithTopic', 'MultipleChoiceTemplatesInstructionsWithoutTopicFixed',
                'MultipleChoiceTemplatesInstructionsWithTopicHelm',
                'MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed',
                'MultipleChoiceTemplatesInstructionsWithoutTopicHarness', 'MultipleChoiceTemplatesStructuredWithTopic',
                'MultipleChoiceTemplatesStructuredWithoutTopic', 'MultipleChoiceTemplatesInstructionsProSASimple',
                'MultipleChoiceTemplatesInstructionsProSALetter', 'MultipleChoiceTemplatesInstructionsProSACould',
                'MultipleChoiceTemplatesInstructionsStateHere', 'MultipleChoiceTemplatesInstructionsStateBelow',
                'MultipleChoiceTemplatesInstructionsStateBelowPlease']
    if dataset_name in ["HellaSwag"]:
        return \
            ['MultipleChoiceTemplatesInstructionsStandard', 'MultipleChoiceTemplatesInstructionsContext',
             'MultipleChoiceTemplatesInstructionsStructured', 'MultipleChoiceTemplatesInstructionsBasic',
             'MultipleChoiceTemplatesInstructionsState1', 'MultipleChoiceTemplatesInstructionsState2',
             'MultipleChoiceTemplatesInstructionsState3', 'MultipleChoiceTemplatesInstructionsState4',
             'MultipleChoiceTemplatesInstructionsState5', 'MultipleChoiceTemplatesInstructionsState6',
             'MultipleChoiceTemplatesInstructionsState7', 'MultipleChoiceTemplatesInstructionsState8']


def run_experiment(local_catalog_path: str):
    """Run the experiment with given configuration"""
    os.environ["UNITXT_ARTIFACTORIES"] = local_catalog_path

    # Generate all possible template combinations
    # Get configurations
    datasets = BaseConfig.DatasetNames.get_datasets()
    subsets = BaseConfig.DatasetSubsets.get_subsets()

    # Generate experiments for each dataset type
    for dataset_name in ["mmlu", "mmlu_pro"]:
        prompt_paraphrases = get_prompt_paraphrasing(dataset_name)
        templates = BaseConfig.PromptOptions.generate_template_combinations(dataset_name)
        template_names = list(templates.keys())
        for prompt_paraphrase in prompt_paraphrases:
            for few_shots in BaseConfig.FewShot.get_values():
                unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
                    "Knowledge": [
                        _DefaultUnitxtRecipeArgs(
                            card=f"cards.{dataset_name}.{subset}",
                            template=[
                                f"{prompt_paraphrase}.{template_name}"
                                for template_name in template_names
                            ],
                            demos_pool_size=few_shots,
                            max_test_instances=100,
                            max_train_instances=100,
                            max_validation_instances=100,
                        )
                        for subset in subsets[dataset_name]
                    ]
                }
                # Your experiment execution code here
                # process_experiment(unitxt_recipe_args_by_groupings)
    for dataset_name in ["ai2_arc.arc_easy",
                         "ai2_arc.arc_challenge",
                         "hellaswag",
                         "openbook_qa",
                         "social_iqa"]:
        catalog_dataset_map = {"ai2_arc.arc_easy": "AI2_ARC",
                             "ai2_arc.arc_challenge": "AI2_ARC",
                             "hellaswag": "HellaSwag",
                             "openbook_qa": "OpenBookQA",
                             "social_iqa": "Social_IQa"}
        prompt_paraphrases = get_prompt_paraphrasing(dataset_name)
        templates = BaseConfig.PromptOptions.generate_template_combinations(dataset_name)
        template_names = list(templates.keys())
        for prompt_paraphrase in prompt_paraphrases:
            for few_shots in BaseConfig.FewShot.get_values():
                unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
                    "Knowledge": [
                        _DefaultUnitxtRecipeArgs(
                            card=f"cards.{dataset_name}.{subset}",
                            template=[
                                f"{catalog_dataset_map[dataset_name]}.{prompt_paraphrase}.{template_name}"
                                for template_name in template_names
                            ],
                            demos_pool_size=few_shots,
                            max_test_instances=100,
                            max_train_instances=100,
                            max_validation_instances=100,
                        )
                        for subset in subsets[dataset_name]
                    ]
                }
                # Your experiment execution code here
                # process_experiment(unitxt_recipe_args_by_groupings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multiple choice templates')
    parser.add_argument('--local_catalog_path', type=str, default="local_catalog_path",
                        help='The local catalog path')
    args = parser.parse_args()

    run_experiment(args.local_catalog_path)
