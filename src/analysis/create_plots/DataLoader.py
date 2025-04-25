# data_loader.py
import time

import pandas as pd
import pyarrow.compute as pc
from datasets import load_dataset
from huggingface_hub import HfApi

repo_name = "nlphuji/DOVE_Lite"


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
        self.repo_name = dataset_name
        self.split = split
        self.batch_size = batch_size

    def load_and_process_data(self, model_name, shots=None,
                              datasets=None, template=None, separator=None, enumerator=None, choices_order=None,
                              max_samples=None, drop=True):
        start_time = time.time()
        # print(f"Processing model: {model_name}")
        self.load_data_with_filter(max_samples, drop, model_name, shots, datasets)
        if len(self.dataset) == 0:
            print(f"No data found for model {model_name} and shots {shots}")
            return pd.DataFrame()
        # full_results = self.extract_data(model_name, shots,
        #                                  dataset=datasets, template=template, separator=separator,
        #                                  enumerator=enumerator,
        #                                  choices_order=choices_order)
        # clean_df = self.remove_duplicates(self.dataset.to_pandas())
        # clean_df = clean_df.drop(['quantization', 'closest_answer'], axis=1)
        load_time = time.time()
        print(f"The size of the data after removing duplicates is: {len(clean_df)}")
        print(f"Data loading completed in {load_time - start_time:.2f} seconds")
        return self.dataset.to_pandas()

    def load_data_with_filter(self, max_samples=None, drop=True, model_name=None, shots=None, datasets=None):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries,
        with filtering during loading for better memory management.
        """
        # Load the dataset if not already loaded

        api = HfApi()
        all_files = api.list_repo_files(
            repo_id="nlphuji/DOVE_Lite",
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
                self.dataset = load_dataset(self.repo_name, data_files=existing_files, split=self.split,
                                            cache_dir=None)
            else:
                self.dataset = load_dataset(self.repo_name, split=self.split)
        print("The size of the data after filtering is: ", len(self.dataset))
        if drop:
            self.dataset = self.dataset.remove_columns(['cumulative_logprob', 'generated_text', 'ground_truth'])

    def extract_data(self, model_name, shots, dataset=None, template=None, separator=None, enumerator=None,
                     choices_order=None):
        arrow_table = self.dataset.data.table

        conditions = [
            pc.equal(arrow_table['model'], model_name),
            pc.equal(arrow_table['shots'], shots)
        ]

        optional_filters = {
            'template': template,
            'separator': separator,
            'enumerator': enumerator,
            'choices_order': choices_order
        }

        for column, value in optional_filters.items():
            if value is not None:
                conditions.append(pc.equal(arrow_table[column], value))

        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = pc.and_(combined_condition, condition)

        try:
            print("Filtering data...")
            filtered_table = arrow_table.filter(combined_condition)
            df_filtered = filtered_table.to_pandas()
            print(df_filtered['model'].value_counts())
            return df_filtered
        except Exception as e:
            raise Exception(f"Error filtering data: {str(e)}")

    def remove_duplicates(self, df):
        """
        Removes duplicates from the DataFrame.
        """
        cols = ['sample_index', 'model', 'dataset', 'template', 'separator', 'enumerator', 'choices_order', 'shots']
        df_unique = df.drop_duplicates(subset=cols)
        return df_unique
