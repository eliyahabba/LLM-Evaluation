from pathlib import Path


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
