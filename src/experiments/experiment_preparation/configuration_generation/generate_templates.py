import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List
from typing import Tuple

from fm_eval.benchmarks.basic.benchmark_types import (
    GenerationArgs,
    UnitxtRecipeArgs,
    UnitxtSingleRecipeArgs,
)
from fm_eval.benchmarks.basic.benchmarks_definitions.utils.benchmark_function import (
    get_basic_benchmark_function,
)


@dataclass
class _DefaultUnitxtRecipeArgs(UnitxtRecipeArgs):
    demos_pool_size: int = 100
    num_demos: List[int] = field(default_factory=lambda: [0, 5])
    system_prompt: List[str] = field(default_factory=lambda: ["system_prompts.empty"])
    format: List[str] = field(default_factory=lambda: ["formats.chat_api"])
    augmentor: List[str] = field(default_factory=lambda: [])
    max_test_instances: int = 100
    max_train_instances = 100
    max_validation_instances = 100


@dataclass
class _DefaultGenerationArgs(GenerationArgs):
    max_new_tokens: int = 64
    seed: List[int] = field(default_factory=lambda: [42])
    top_p: List[float] = field(default_factory=lambda: [])
    top_k: List[int] = field(default_factory=lambda: [])
    temperature: List[float] = field(default_factory=lambda: [])
    do_sample: bool = False
    num_beams: int = 1
    stop_sequences: List[List[str]] = field(default_factory=lambda: [])
    quantization: List[str] = field(default_factory=lambda: ["auto"])
    max_predict_samples: int = 100


class MultiChoiceDatasetsConfig:
    """Dataset subset configurations"""

    @staticmethod
    def get_card_list(dataset_name: str) -> List[str]:
        dataset_subset_dict = {
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
            ],
            "ai2_arc.arc_easy":
            [],
            "ai2_arc.arc_challenge":
            [],
            "hellaswag":
            [],
            "openbook_qa":
            [],
            "social_iqa":
            []

        }

        subset_list = dataset_subset_dict.get(dataset_name, [])
        if not subset_list:
            return [f"cards.{dataset_name}"]
        cards_list = []
        for subset in subset_list:
            cards_list.append(f"cards.{dataset_name}.{subset}")
        return cards_list


def get_run_data(dataset_name: str) -> List[Tuple[str, List[str]]]:
    with open('experiments_config.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)

    dataset_config = configs[dataset_name]
    results = []
    dataset_catalog_name = dataset_config["dataset_catalog_name"]
    prompt_paraphrases = dataset_config["prompt_paraphrases"]
    template_names = dataset_config["template_names"]
    for subset in MultiChoiceDatasetsConfig.get_card_list(dataset_name):
        card_name = f"cards.{dataset_name}.{subset}"
        templates = [f"templates.huji_workshop.{dataset_catalog_name}.{prompt}.{template}"
                     for prompt in prompt_paraphrases
                     for template in template_names]
        results.append((card_name, templates))
    return results



def run_experiment(local_catalog_path: str):
    """Run the experiment with given configuration"""
    os.environ["UNITXT_ARTIFACTORIES"] = local_catalog_path

    # Generate all possible template combinations
    configs = ["mmlu", "mmlu_pro", "ai2_arc.arc_easy", "ai2_arc.arc_challenge", "hellaswag", "openbook_qa",
               "social_iqa"]
    for config in configs:
        unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
            config: [
                _DefaultUnitxtRecipeArgs(
                    card=card_name,
                    template=templates
                )
                for card_name, templates in get_run_data(config)
            ]
        }

        def get_generation_args(unitxt_args: UnitxtSingleRecipeArgs) -> GenerationArgs:
            return _DefaultGenerationArgs()

        get_single_runs_args_list = get_basic_benchmark_function(
            unitxt_recipe_args_by_groupings,
            get_run_generation_args_func=get_generation_args,
            get_train_args_func=None,
            system_prompts_and_formatters_mapper=None,
        )
