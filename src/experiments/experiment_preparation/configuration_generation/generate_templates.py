import argparse
import itertools
import os
from enum import Enum
from typing import Dict, List


class BaseConfig:
    """Parent configuration class containing all config-related classes and methods"""

    class DatasetNames:
        """Dataset name constants"""
        MMLU = "mmlu"
        MMLU_PRO = "mmlu_pro"

        @classmethod
        def get_datasets(cls) -> List[str]:
            return [cls.MMLU, cls.MMLU_PRO]

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
        MC_WITHOUT_TOPIC = "MultipleChoiceTemplatesInstructionsWithoutTopic"
        MC_WITH_TOPIC_HELM = "MultipleChoiceTemplatesInstructionsWithTopicHelm"
        MC_WITHOUT_TOPIC_HELM = "MultipleChoiceTemplatesInstructionsWithoutTopicHelm"

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
        FEW_SHOT = 4

        @classmethod
        def get_values(cls) -> List[int]:
            return [shot.value for shot in cls]

    class PromptOptions:
        """Prompt variation options"""
        GREEK_CHARS = "αβγδεζηθικ"

        @classmethod
        def get_options(cls) -> Dict[str, List]:
            return {
                "enumerator": ["capitals", "lowercase", "numbers", "roman", "!@#$%^₪*)(", cls.GREEK_CHARS],
                "choices_separator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
                "shuffle_choices": [False, True],
            }

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
    def generate_template_combinations(cls) -> List[Dict]:
        """Generate all possible combinations of template options"""
        options = cls.PromptOptions.get_options()
        return [
            {key: value for key, value in zip(options.keys(), values)}
            for values in itertools.product(*options.values())
        ]


def run_experiment(local_catalog_path: str):
    """Run the experiment with given configuration"""
    os.environ["UNITXT_ARTIFACTORIES"] = local_catalog_path

    # Generate all possible template combinations
    templates = BaseConfig.generate_template_combinations()
    template_names = [f"template_{i}" for i in range(len(templates))]

    # Get configurations
    datasets = BaseConfig.DatasetNames.get_datasets()
    subsets = BaseConfig.DatasetSubsets.get_subsets()

    # Generate experiments for each dataset type
    for dataset_name in datasets:
        for prompt_template in BaseConfig.PromptParaphrases.get_all_prompt_paraphrases():
            for few_shots in BaseConfig.FewShot.get_values():
                unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
                    "Knowledge": [
                        _DefaultUnitxtRecipeArgs(
                            card=f"cards.{dataset_name}.{subset}",
                            template=[
                                f"{prompt_template}.{template_name}"
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
