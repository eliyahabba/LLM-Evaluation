import itertools
import os
from typing import Dict, List

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class TemplateConfig:
    OVERRIDE_OPTIONS = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman"],
        "choices_separator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
        "shuffle_choices": [False, True],
    }

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
    def priority_models_order():
        return [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct"
        ]

    @staticmethod
    def priority_datasets_order():
        return [
            "mmlu_pro",
            "mmlu",
        ]


if __name__ == "__main__":
    ############################################################################################################
    # (just for reference, there are already defined in the MultipleChoiceTemplateGenerator class)
    possible_templates = [
        {key: value for key, value in zip(TemplateConfig.OVERRIDE_OPTIONS.keys(), values)}
        for values in itertools.product(*TemplateConfig.OVERRIDE_OPTIONS.values())
    ]
    ############################################################################################################


    # This is peace of code that we need to copy for running the experiment
    templates_names = [f"template_{i}" for i in range(len(possible_templates))]

    # Define the local catalog path
    multiple_choice_instructions_with_topic_folder_name_local_catalog_path = TemplatesGeneratorConstants.DATA_PATH / TemplatesGeneratorConstants.MULTIPLE_CHOICE_INSTRUCTIONS_WITH_TOPIC_FOLDER_NAME
    # each catalog path contains different prompt paraphrase
    os.environ["UNITXT_ARTIFACTORIES"] = f"{str(multiple_choice_instructions_with_topic_folder_name_local_catalog_path)}"

    # Define the configurations for the experiment
    unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
        "Knowledge": [
            _DefaultUnitxtRecipeArgs(
                card=f"cards.mmlu_pro.{subset}",
                template=[f"mmlu_pro.{subset}.{templates_name}" for templates_name in templates_names],
                demos_pool_size=0,
                max_test_instances=100,
                max_train_instances=100,
                max_validation_instances=100,
            )
            for x in TemplateConfig.priority_datasets_order() for subset in TemplateConfig.SUBSETS[x]
        ]
    }
