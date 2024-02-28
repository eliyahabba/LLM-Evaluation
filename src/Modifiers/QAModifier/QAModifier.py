from pathlib import Path

from src.Modifiers.DatasetModifier import DatasetModifier
from src.utils.Constants import Constants

DatasetModifierConstants = Constants.DatasetModifier
SocialQAConstants = Constants.SocialQAModifier

class QAModifier(DatasetModifier):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def modify_dataset(self):
        # Implement modifications specific to QA dataset
        pass  # Implement QA modifications here

    def paraphrase_question(self, question, paraphraser_model):
        # Implement question paraphrasing
        pass
