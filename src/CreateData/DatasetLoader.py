from unitxt.standard import StandardRecipe
from unitxt.templates import Template

from src.CreateData.LLMDataset import LLMDataset


class DatasetLoader:
    def __init__(self, card: str, template: Template, system_format: str,
                 max_instances: int, template_name: str):
        self.card = card
        self.template = template
        self.system_format = system_format
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
            max_train_instances=self.max_instances,
            max_validation_instances=self.max_instances,
            max_test_instances=self.max_instances,
        )

        dataset = recipe().to_dataset()

        llm_dataset = LLMDataset(self.template_name, dataset)
        return llm_dataset
