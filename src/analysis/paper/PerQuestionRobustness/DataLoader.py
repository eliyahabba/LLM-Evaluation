# data_loader.py
import time

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

repo_name = "eliyahabba/llm-evaluation-analysis-split"


class DataLoader:
    def __init__(self, dataset_name=repo_name, split="train", batch_size=10000):
        """
        Initializes the DataLoader with dataset details.

        Args:
            dataset_name (str): Name of the dataset to load from HuggingFace.
            split (str): Dataset split to use.
            batch_size (int): Number of samples per batch.
        """
        self.dataset = None
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size

    def load_and_process_data(self, model_name, shots,
                              datasets=None, drop=True):
        start_time = time.time()
        self.load_data_with_filter(drop, model_name, shots, datasets)
        if len(self.dataset) == 0:
            print(f"No data found for model {model_name} and shots {shots}")
            return pd.DataFrame()
        clean_df = self.remove_duplicates()
        clean_df = clean_df.drop(['quantization', 'closest_answer'], axis=1)
        load_time = time.time()
        print(f"The size of the data after removing duplicates is: {len(clean_df)}")
        print(f"Data loading completed in {load_time - start_time:.2f} seconds")
        return clean_df

    def load_data_with_filter(self, drop=True, model_name=None, shots=None, datasets=None):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries,
        with filtering during loading for better memory management.
        """
        # Load the dataset if not already loaded

        api = HfApi()
        all_files = api.list_repo_files(
            repo_id="eliyahabba/llm-evaluation-analysis-split",
            repo_type="dataset",
        )
        existing_files = [file for file in all_files if file.endswith('.parquet')]
        # split only the file name that contains the model name and shots and dataset
        if model_name is not None:
            model_name = model_name.split("/")[-1]
            existing_files = [file for file in existing_files if model_name in file]
        if shots is not None:
            shots = "shots" + str(shots)
            existing_files = [file for file in existing_files if shots in file]
        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]
            datasets = [dataset.split("/")[-1] for dataset in datasets]
            existing_files = [file for file in existing_files if any(dataset in file for dataset in datasets)]

        if len(existing_files) > 0:
            if self.dataset is None:
                # split = split_with_filter
                self.dataset = load_dataset(self.dataset_name, data_files=existing_files, split=self.split)
            else:
                self.dataset = load_dataset(self.dataset_name, split=self.split)
        print("The size of the data after filtering is: ", len(self.dataset))
        if drop:
            self.dataset = self.dataset.remove_columns(['cumulative_logprob', 'generated_text', 'ground_truth'])
        return self.dataset

    def remove_duplicates(self):
        """
        Removes duplicates from the DataFrame.
        """
        cols = ['sample_index', 'model', 'dataset', 'template', 'separator', 'enumerator', 'choices_order', 'shots']
        df_unique = self.dataset.drop_duplicates(subset=cols)
        return df_unique
