from pathlib import Path


class DatasetModifier:
    """
    Base class for dataset modifiers. Defines the interface for modification.
    """

    def __init__(self, data_path: Path, modified_data_path: Path):
        self.data_path = data_path
        self.modified_data_path = modified_data_path

    def load_data(self):
        """
        Loads data from the specified path. Can be overridden if needed.
        """
        with open(self.data_path, "r") as f:
            return f.read()

    def save_data(self, data) -> None:
        """
        Saves data to the specified path. Can be overridden if needed.
        """
        with open(self.modified_data_path, "w") as f:
            f.write(data)

    def modify(self):
        """
        Modifies the dataset and saves it to the specified path.
        Needs to be implemented by subclasses.
        """
        data = self.load_data()
        modified_data = self._modify(data)
        self.save_data(modified_data)

    def _modify(self, data: str) -> str:
        """
        Modifies the dataset. Needs to be implemented by subclasses.
        :param data: The dataset
        :return: str: The modified dataset
        """
        raise NotImplementedError()
