from unitxt.standard import StandardRecipe

from src.CreateData.LLMDataset import LLMDataset


class DatasetLoader:
    def __init__(self, card: str, template: str, num_demos: int, demos_pool_size: int, system_format: str,
                 max_train_instances: int, template_name: str):
        self.card = card
        self.template = template
        self.num_demos = num_demos
        self.demos_pool_size = demos_pool_size
        self.system_format = system_format
        self.max_train_instances = max_train_instances
        self.template_name = template_name

    def load(self) -> LLMDataset:
        """
        Loads the dataset from the specified path.

        @return: The dataset
        """
        recipe = StandardRecipe(
            card=self.card,
            template=self.template,
            num_demos=self.num_demos,
            demos_pool_size=self.demos_pool_size,
            format=self.system_format,
            max_train_instances=self.max_train_instances
        )

        dataset = recipe().to_dataset()

        llm_dataset = LLMDataset(self.template_name, dataset)
        return llm_dataset
