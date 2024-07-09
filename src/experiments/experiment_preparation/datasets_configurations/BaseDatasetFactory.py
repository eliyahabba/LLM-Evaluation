class BaseDatasetFactory:
    """
    Base class for dataset factories.
    """
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def get_dataset(self):
        """
        Returns the dataset for the given dataset name.
        """
        raise NotImplementedError("Subclasses must implement this method to return the specific dataset.")

    def create_dataset(self):
        """
        Creates the dataset for the given dataset name.
        """
        raise NotImplementedError("Subclasses must implement this method to create the specific dataset.")