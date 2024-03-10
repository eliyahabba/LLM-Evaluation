from abc import abstractmethod

from src.Modifiers.DatasetModifier import DatasetModifier
from src.utils.Constants import Constants

DatasetModifierConstants = Constants.DatasetModifierConstants


class QAModifier(DatasetModifier):
    @abstractmethod
    def _modify(self, data: str) -> str:
        """
        Modifies the dataset. Needs to be implemented by subclasses.
        @param data: The dataset
        @return: str: The modified dataset
        """
        raise NotImplementedError()

    def paraphrase_question(self, question:str, paraphraser_model: str) -> str:
        """
        Paraphrases the question using the specified model
        @param question: The question
        @param paraphraser_model: The name / path of model to use for paraphrasing
        """
        # Implement question paraphrasing
        pass
