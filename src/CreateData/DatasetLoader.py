from unitxt.standard import StandardRecipe
from unitxt.templates import Template

from src.CreateData.LLMDataset import LLMDataset


class DatasetLoader:
    def __init__(self, card: str, template: Template, system_format: str, num_demos: int, demos_pool_size: int,
                 max_instances: int, template_name: str):
        self.card = card
        self.template = template
        self.system_format = system_format
        self.num_demos = num_demos
        self.demos_pool_size = demos_pool_size
        self.max_instances = max_instances
        self.template_name = template_name

    def load(self) -> LLMDataset:
        """
        Loads the dataset from the specified path.

        @return: The dataset
        """
        recipe = StandardRecipe(
            card=self.card,
            # template="templates.qa.multiple_choice.with_context.no_intro.helm[enumerator=[option 1, option 2]]",
            template=self.template,
            format=self.system_format,
            num_demos=self.num_demos,
            demos_pool_size=self.demos_pool_size,
            max_train_instances=self.max_instances,
            max_validation_instances=self.max_instances,
            max_test_instances=self.max_instances,
        )

        dataset = recipe().to_dataset()

        llm_dataset = LLMDataset(self.template_name, dataset)
        return llm_dataset
