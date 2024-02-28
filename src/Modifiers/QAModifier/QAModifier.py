from pathlib import Path

from src.Modifiers.DatasetModifier import DatasetModifier
from src.utils.Constants import Constants

from abc import abstractmethod
DatasetModifierConstants = Constants.DatasetModifier

class QAModifier(DatasetModifier):
    @abstractmethod
    def _modify(self, data: str) -> str:
        """
        Modifies the dataset. Needs to be implemented by subclasses.
        :param data: The dataset
        :return: str: The modified dataset
        """
        raise NotImplementedError()

    def paraphrase_question(self, question, paraphraser_model):
        # Implement question paraphrasing
        pass
