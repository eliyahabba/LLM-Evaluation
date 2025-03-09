from pathlib import Path

import pandas as pd

from config.get_config import Config
from src.experiments.experiment_preparation.configuration_generation.TemplateVariationDimensions import TemplateVariationDimensions
from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class Utils:
    @staticmethod
    def get_card_name(card: str) -> str:
        """
        Get the name of the card.
        @param card: The card name
        @return: The name of the card
        """
        return card.split('cards.')[1]

    @staticmethod
    def get_card_path(path: Path, card: str) -> Path:
        """
        Get the path of the card.
        @param path: The path of templates or results folder
        @param card: The card name
        @return: The path of the card
        """
        return path / Utils.get_card_name(card)

    @staticmethod
    def get_num_from_template_name(template_name: str) -> int:
        """
        Get the name of the template.
        @param template_name: The name of the template
        @return: The number of the template
        """
        return int(template_name.split("_")[-1])

    @staticmethod
    def get_template_name(template_num: int) -> str:
        """
        Get the name of the template.
        @param template_num: The number of the template
        @return: The name of the template
        """
        templates_metadata = pd.read_csv(TemplatesGeneratorConstants.TEMPLATES_METADATA_PATH, index_col='template_name')
        template = templates_metadata.loc[f"template_{template_num}"]
        # convert template to dictionary
        template_dict = template.to_dict()
        template_name = TemplateVariationDimensions.generate_template_name(template_dict)
        return template_name

    @staticmethod
    def get_system_format_class(system_format: str) -> str:
        """
        Get the system format class.
        @param system_format: The system format
        @return: The system format class
        """
        return ExperimentConstants.SYSTEM_FORMATS_NAMES[system_format]

    @staticmethod
    def get_access_token() -> str:
        """
        Get the access token.
        @return:
        """
        config = Config()
        access_token = config.config_values.get("access_token", "")
        return access_token

    @staticmethod
    def get_model_name(model_name) -> str:
        """
        Get the model name from the full and official model name.
        @param model_name: The full and official model name.
        @return: The model name.
        """
        return model_name.split('/')[-1]

    @staticmethod
    def word_to_number(word):
        """
        Convert a number word to its integer value.
        Raises ValueError if the word is not a valid number word.
        """
        number_dict = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10
        }

        # Convert input to lowercase for case-insensitive matching
        word = word.lower()

        # Check if the word exists in our dictionary
        if word in number_dict:
            return number_dict[word]
        else:
            raise ValueError(f"Invalid number word: {word}")
