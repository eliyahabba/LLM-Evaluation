from abc import abstractmethod

from src.Modifiers.DatasetModifier import DatasetModifier
from src.utils.Constants import Constants

DatasetModifierConstants = Constants.DatasetModifier


class MultipleChoiceModifier(DatasetModifier):
    """
    abstract class for multiple choice modifiers
    """
    @abstractmethod
    def _modify(self, data: str) -> str:
        """
        Modifies the dataset. Needs to be implemented by subclasses.
        :param data: The dataset
        :return: str: The modified dataset
        """
        raise NotImplementedError()

    def shuffle_answers(answers: List[str], gold_answer_index: int) -> Tuple[List[str], int]:
        """
        Shuffles the answers and returns the shuffled answers and the index of the gold answer
        :param answers: The list of answers
        :param gold_answer_index: The index of the gold answer
        :return: Tuple[List[str], int]: The shuffled answers and the index of the gold answer
        """
        return answers, gold_answer_index



if __name__ == "__main__":
    pass