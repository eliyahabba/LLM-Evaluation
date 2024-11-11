import os
from typing import Dict, List

from src.utils.Constants import Constants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
local_catalog_path = TemplatesGeneratorConstants.DATA_PATH / TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_TOPIC_FOLDER_NAME
os.environ["UNITXT_ARTIFACTORIES"] = str(local_catalog_path)

subsets = {  # the key must appear in the card name
    "mmlu_pro": [
        "history",
        "law",
        "health",
        "physics",
        "business",
        "other",
        "philosophy",
        "psychology",
        "economics",
        "math",
        "biology",
        "chemistry",
        "computer_science",
        "engineering",
    ],
    "mmlu": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions"
    ]
}

templates_nums = [f"template_{i}" for i in range(0, 56)]

def priority_order():
    return [
        "mmlu_pro",
        "mmlu",
    ]

unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
    "Knowledge": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.mmlu_pro.{subset}",
            template=[f"mmlu_pro.{subset}.{template_num}" for template_num in templates_nums],
            demos_pool_size=0,
            demos_taken_from="validation",
        )
        for x in priority_order() for subset in subsets[x]
    ]
}