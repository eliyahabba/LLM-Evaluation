import os

import numpy as np
import pandas as pd
from unitxt.formats import SystemFormat
from unitxt.standard import StandardRecipe
from unitxt.templates import Template, MultipleChoiceTemplate

from src.experiments.data_loading.NLPDataset import NLPDataset
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


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

    def read_dataset_sizes(self):
        """
        Reads the MMLU dataset sizes from the file.

        @return: The MMLU dataset sizes
        """
        dataset_sizes = pd.read_csv(TemplatesGeneratorConstants.DATASET_SIZES_PATH)
        return dataset_sizes

    def get_validation_size(self, card: str):
        """
        Gets the validation size for the specified card.

        @param card: The card
        @return: The validation size
        """
        dataset_sizes = self.read_dataset_sizes()
        validation_size = \
            dataset_sizes[dataset_sizes["Name"] == card.split("cards.")[1]]["validation"].values[0]
        return validation_size

    def load(self) -> NLPDataset:
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
        if self.num_demos:
            if not np.isnan(self.get_validation_size(self.card)):
                self.demos_pool_size = self.get_validation_size(self.card) - 1
            else:
                self.demos_taken_from = 'train'


        recipe = StandardRecipe(
            card=self.card,
            template=self.template,
            format=system_format,
            demos_taken_from=self.demos_taken_from,
            num_demos=self.num_demos,
            demos_pool_size=None if "mmlu" in self.card and self.demos_pool_size == 0 else self.demos_pool_size,
            max_train_instances=self.max_instances,
            max_validation_instances=self.max_instances,
            max_test_instances=self.max_instances,
        )

        dataset = recipe().to_dataset()

        llm_dataset = NLPDataset(self.template_name, dataset)
        return llm_dataset


if __name__ == '__main__':
    os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"
    topic = "Topic_1"
    question = "What is the capital of France?"
    choices = ["Paris, London, Berlin, Rome"]

    input_format=(f"Question: [question] Choices: [choices] Answer:"
                  f" [Answer]\nChoices: {choices} Answer:)")
    # input_format=(f"Question")
    template = MultipleChoiceTemplate(input_format=input_format,
    target_prefix="",
    choices_field= "choices",
    target_field = "answer",
    choices_separator= ", ",
    source_choice_format = "{choice_numeral}. {choice_text}",
    target_choice_format = "{choice_numeral}",
    enumerator = "capitals",
    shuffle_choices= False
    )
    loader = DatasetLoader("cards.mmlu_pro.biology", template, "empty_system_format",
                           "train", 1, 2, 1000, "mmlu")
    dataset = loader.load()

