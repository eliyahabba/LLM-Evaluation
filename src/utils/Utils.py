from pathlib import Path


class Utils:
    @staticmethod
    def get_card_path(path: Path, card: str) -> Path:
        """
        Get the path of the card.
        @param path: The path of templates or results folder
        @param card: The card name
        @return: The path of the card
        """
        card_name = card.split('cards.')[1]
        return path / card_name
