from pathlib import Path

from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants


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
    def get_template_name(template_num: int) -> str:
        """
        Get the name of the template.
        @param template_num: The number of the template
        @return: The name of the template
        """
        return f"template_{template_num}"

    @staticmethod
    def get_system_format_class(system_format: str) -> str:
        """
        Get the system format class.
        @param system_format: The system format
        @return: The system format class
        """
        # get the  key of ExperimentConstants.SYSTEM_FORMATS from the system_format value
        return list(ExperimentConstants.SYSTEM_FORMATS_NAMES.keys())[
            list(ExperimentConstants.SYSTEM_FORMATS_NAMES.values()).index(system_format)]
