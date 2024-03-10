from datasets import DatasetDict


class LLMDataset:
    def __init__(self, data_name, dataset: DatasetDict):
        self.data_name = data_name
        self.dataset = dataset
