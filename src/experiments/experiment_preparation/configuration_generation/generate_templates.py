import os
from typing import Dict, List

from src.utils.Constants import Constants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
local_catalog_path = TemplatesGeneratorConstants.DATA_PATH / TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_TOPIC_FOLDER_NAME
os.environ["UNITXT_ARTIFACTORIES"] = str(local_catalog_path)

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