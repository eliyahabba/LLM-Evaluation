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
        return ExperimentConstants.SYSTEM_FORMATS_NAMES[system_format]

    @staticmethod
    def get_access_token() -> str:
        """
        Get the access token.
        @return:
        """
        with open(Path(__file__).parent / "access_token", "r") as file:
            return file.readline().strip()

    @staticmethod
    def get_model_name(model_name) -> str:
        """
        Get the model name from the full and official model name.
        @param model_name: The full and official model name.
        @return: The model name.
        """
        return model_name.split('/')[-1]
