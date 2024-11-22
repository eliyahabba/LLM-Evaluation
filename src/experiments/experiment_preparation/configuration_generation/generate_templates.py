import itertools
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
MMLU = "MMLU"
MMLU_PRO = "MMLU_PRO"


class FewShotNum(Enum):
    ZERO_SHOT = 0
    FEW_SHOT = 4  # or whatever other values you want to support


class TemplateConfig:
    prompt_variations = ConfigParams.override_options

    DATASETS = {MMLU: "mmlu", MMLU_PRO: "mmlu_pro"}

    SUBSETS = {
        "mmlu_pro": [
            "history", "law", "health", "physics", "business", "other",
            "philosophy", "psychology", "economics", "math", "biology",
            "chemistry", "computer_science", "engineering",
        ],
        "mmlu": [
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

    @staticmethod
    def get_model_configs():
        return {
            "Llama": {
                "Llama-3.2-1B-Instruct": {
                    "path": "meta-llama/Llama-3.2-1B-Instruct",
                    "size": "1B"
                },
                "Llama-3.2-3B-Instruct": {
                    "path": "meta-llama/Llama-3.2-3B-Instruct",
                    "size": "3B"
                },
                "Meta-Llama-3-8B-Instruct": {
                    "path": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "size": "8B"
                },
                "Meta-Llama-3-70B-Instruct": {
                    "path": "meta-llama/Meta-Llama-3-70B-Instruct",
                    "size": "70B"
                }
            },

            "Mistral": {
                "Mistral-7B-Instruct-v0.3": {
                    "path": "mistralai/Mistral-7B-Instruct-v0.3",
                    "size": "7B"
                },
                "Mixtral-8x7B-Instruct-v0.1": {
                    "path": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "size": "8x7B"
                },
                "Mixtral-8x22B-Instruct-v0.1": {
                    "path": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                    "size": "8x22B"
                }
            },

            "OLMo": {
                "OLMo-7B-Instruct": {
                    "path": "allenai/OLMo-7B-Instruct",
                    "size": "7B"
                },
                "OLMoE-1B-7B-0924-Instruct": {
                    "path": "allenai/allenai/OLMoE-1B-7B-0924-Instruct",
                    "size": "1B"
                }
            },

            "Qwen": {
                "Qwen2.5-0.5B-Instruct": {
                    "path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "size": "0.5B"
                },
                "Qwen2.5-3B-Instruct": {
                    "path": "Qwen/Qwen2.5-3B-Instruct",
                    "size": "3B"
                },
                "Qwen2.5-7B-Instruct": {
                    "path": "Qwen/Qwen2.5-7B-Instruct",
                    "size": "7B"
                },
                "Qwen2.5-72B-Instruct": {
                    "path": "Qwen/Qwen2.5-72B-Instruct",
                    "size": "72B"
                }
            }
        }

    @staticmethod
    def priority_models_order():
        model_configs = TemplateConfig.get_model_configs()
        order = ["Llama", "Mistral", "OLMo", "Gemma", "Qwen"]
        prioritized_paths = []
        for family in order:
            if family in model_configs:
                family_paths = [model_info["path"] for model_info in model_configs[family].values()]
                prioritized_paths.extend(family_paths)
        return prioritized_paths

    @staticmethod
    def priority_few_shots_params():
        return [shot.value for shot in FewShotNum]

    @staticmethod
    def priority_prompt_paraphrase_params() -> List[Path]:
        return [
            TemplatesGeneratorConstants.DATA_PATH / TemplatesGeneratorConstants.MULTIPLE_CHOICE_INSTRUCTIONS_WITH_TOPIC_FOLDER_NAME,
            TemplatesGeneratorConstants.DATA_PATH / TemplatesGeneratorConstants.MULTIPLE_CHOICE_INSTRUCTIONS_WITHOUT_TOPIC_FOLDER_NAME
        ]


if __name__ == "__main__":
    ############################################################################################################
    # (just for reference, there are already defined in the MultipleChoiceTemplateGenerator class)
    possible_templates = [
        {key: value for key, value in zip(TemplateConfig.prompt_variations.keys(), values)}
        for values in itertools.product(*TemplateConfig.prompt_variations.values())
    ]
    ############################################################################################################
    # This is a piece of code that we need to copy to run the experiment
    templates_names = [f"template_{i}" for i in range(len(possible_templates))]

    ## MMLU Pro
    # Select the prompt paraphrase
    for local_prompt_path in TemplateConfig.priority_prompt_paraphrase_params():
        # Define the local catalog path
        os.environ["UNITXT_ARTIFACTORIES"] = f"{str(local_prompt_path)}"
        # Select the few shots value
        for few_shots_value in TemplateConfig.priority_few_shots_params():
            # Generate args for each dataset and its subsets
            unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
                "Knowledge": [
                    _DefaultUnitxtRecipeArgs(
                        card=f"cards.{TemplateConfig.DATASETS[MMLU_PRO]}.{subset}",
                        template=[f"{TemplateConfig.DATASETS[MMLU_PRO]}.{subset}.{templates_name}" for templates_name in
                                  templates_names],
                        demos_pool_size=few_shots_value,
                        max_test_instances=100,
                        max_train_instances=100,
                        max_validation_instances=100,
                    )
                    for subset in TemplateConfig.SUBSETS[TemplateConfig.DATASETS[MMLU_PRO]]
                ]
            }
    ## MMLU
    # Select the prompt paraphrase
    for local_prompt_path in TemplateConfig.priority_prompt_paraphrase_params():
        # Define the local catalog path
        os.environ["UNITXT_ARTIFACTORIES"] = f"{str(local_prompt_path)}"
        # Select the few shots value
        # TODO: Is unixt known to take the template from the new catalog?
        for few_shots_value in TemplateConfig.priority_few_shots_params():
            # Generate args for each dataset and its subsets
            unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
                "Knowledge": [
                    _DefaultUnitxtRecipeArgs(
                        card=f"cards.{TemplateConfig.DATASETS[MMLU]}.{subset}",
                        template=[f"{TemplateConfig.DATASETS[MMLU]}.{subset}.{templates_name}" for templates_name in
                                  templates_names],
                        demos_pool_size=few_shots_value,
                        max_test_instances=100,
                        max_train_instances=100,
                        max_validation_instances=100,
                    )
                    for subset in TemplateConfig.SUBSETS[TemplateConfig.DATASETS[MMLU]]
                ]
            }
