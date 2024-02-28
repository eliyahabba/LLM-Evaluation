from pathlib import Path

from src.Modifiers.DatasetModifier import DatasetModifier

from src.utils.Constants import Constants

DatasetModifierConstants = Constants.DatasetModifier
SocialQAConstants = Constants.SocialQAModifier


class SocialQAModifier(DatasetModifier):
    """
    Modifier for SocialQA dataset.
    """

    def __init__(self, data_path: Path, modified_data_path: Path):
        super().__init__(data_path, modified_data_path)

    def _modify(self, data: str) -> str:
        """
        Modifies the dataset (a json file)
        :param data: The dataset
        :return: str: The modified dataset
        """
        pass
        return data


if __name__ == "__main__":
    data_path = DatasetModifierConstants.DATA_PATH / SocialQAConstants.DATA_NAME / DatasetModifierConstants.TEST_FILE
    modified_data_path = DatasetModifierConstants.MODIFIED_DATA_PATH / SocialQAConstants.DATA_NAME / DatasetModifierConstants.TEST_FILE
    social_qa_modifier = SocialQAModifier(data_path, modified_data_path)
    social_qa_modifier.modify()
