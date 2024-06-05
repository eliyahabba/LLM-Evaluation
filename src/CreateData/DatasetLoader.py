from unitxt.formats import SystemFormat
from unitxt.standard import StandardRecipe
from unitxt.templates import Template

from src.CreateData.LLMDataset import LLMDataset


class DatasetLoader:
    def __init__(self, card: str, template: Template, system_format: str, demos_taken_from: str, num_demos: int,
                 demos_pool_size: int,
                 max_instances: int, template_name: str):
        self.card = card
        self.template = template
        self.system_format = system_format
        self.demos_taken_from = demos_taken_from
        self.num_demos = num_demos
        self.demos_pool_size = demos_pool_size
        self.max_instances = max_instances
        self.template_name = template_name

    def load(self) -> LLMDataset:
        """
        Loads the dataset from the specified path.

        @return: The dataset
        """
        if self.system_format.startswith("formats."):
            system_format = self.system_format
        else:
            system_format = SystemFormat(
                model_input_format=f"{self.system_format}\n{{source}}",
            )
        recipe = StandardRecipe(
            card=self.card,
            template=self.template,
            format=system_format,
            demos_taken_from=self.demos_taken_from,
            num_demos=self.num_demos,
            demos_pool_size=self.demos_pool_size if "mmlu" not in self.card else None,
            max_train_instances=self.max_instances,
            max_validation_instances=self.max_instances,
            max_test_instances=self.max_instances,
        )

        dataset = recipe().to_dataset()

        llm_dataset = LLMDataset(self.template_name, dataset)
        return llm_dataset
