from pathlib import Path

from src.Modifiers.MultipleChoiceModifier.MultipleChoiceModifier import MultipleChoiceModifier
from src.utils.Constants import Constants

DatasetModifierConstants = Constants.DatasetModifierConstants
MMLUModifierConstants = Constants.MMLUModifierConstants


class MMLUModifier(MultipleChoiceModifier):
    """
    Modifier for MMLU dataset.
    """

    def __init__(self, data_path: Path, modified_data_path: Path):
        super().__init__(data_path, modified_data_path)

    def _modify(self, data: str) -> str:
        """
        Modifies the dataset (a json file)
        @param data: The dataset
        @return: str: The modified dataset
        """
        pass
        return data


if __name__ == "__main__":
    data_path = DatasetModifierConstants.DATA_PATH / MMLUModifierConstants.DATA_NAME / DatasetModifierConstants.TEST_FILE
    modified_data_path = DatasetModifierConstants.MODIFIED_DATA_PATH / MMLUModifierConstants.DATA_NAME / DatasetModifierConstants.TEST_FILE
    mmlu_modifier = MMLUModifier(data_path, modified_data_path)
    mmlu_modifier.modify()
