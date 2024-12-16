from pathlib import Path

import pandas as pd

from config.get_config import Config
from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
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
        template_name = ConfigParams.generate_template_name(template_dict)
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
