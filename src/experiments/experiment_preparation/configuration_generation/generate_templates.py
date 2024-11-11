import os
from dataclasses import dataclass, field
from typing import Dict, List

from fm_eval.benchmarks.basic.benchmark_types import (
    GenerationArgs,
    UnitxtRecipeArgs,
    UnitxtSingleRecipeArgs,
)
from fm_eval.benchmarks.basic.benchmarks_definitions.utils.benchmark_function import (
    get_basic_benchmark_function,
)

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
    ]
}


@dataclass
class _DefaultUnitxtRecipeArgs(UnitxtRecipeArgs):
    demos_pool_size: int = 100
    num_demos: List[int] = field(default_factory=lambda: [0, 5])
    system_prompt: List[str] = field(default_factory=lambda: ["system_prompts.empty"])
    format: List[str] = field(default_factory=lambda: ["formats.llama3_instruct"])
    max_train_instances: int = 1000
    max_validation_instances: int = 1000


@dataclass
class _DefaultGenerationArgs(GenerationArgs):
    max_new_tokens: int = 64
    seed: List[int] = field(default_factory=lambda: [42])
    top_p: List[float] = field(default_factory=lambda: [])
    top_k: List[int] = field(default_factory=lambda: [])
    temperature: List[float] = field(default_factory=lambda: [])
    do_sample: bool = False
    num_beams: int = 1
    stop_sequences: List[List[str]] = field(default_factory=lambda: [["\n\n"]])
    max_predict_samples: int = 100



templates_nums = [f"template_{i}" for i in range(0, 56)]
unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
    "Knowledge": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.mmlu_pro.{subset}",
            template=[f"mmlu_pro.{subset}.{template_num}" for template_num in templates_nums],
            demos_pool_size=20,
            demos_taken_from="test",
        )
        for subset in subsets["mmlu_pro"]
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

