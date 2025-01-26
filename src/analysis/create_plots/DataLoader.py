# data_loader.py
import time

import pandas as pd
import pyarrow.compute as pc
from datasets import load_dataset


class DataLoader:
    def __init__(self, dataset_name="eliyahabba/llm-evaluation-analysis", split="train", batch_size=10000):
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

    def load_and_process_data_batched(self, model_name, shots, max_samples=None):
        start_time = time.time()
        # print(f"Processing model: {model_name}")
        partial_results = []
        for batch_result in self.load_data_batched(model_name, shots, max_samples):
            partial_results.append(batch_result)
        df_partial = pd.concat(partial_results, ignore_index=True)
        clean_df = self.remove_duplicates(df_partial)
        load_time = time.time()
        print(f"Data loading completed in {load_time - start_time:.2f} seconds")
        return clean_df

    def load_and_process_data(self, model_name, shots,
                              dataset=None, template=None, separator=None, enumerator=None, choices_order=None,
                              max_samples=None, drop=True):
        start_time = time.time()
        # print(f"Processing model: {model_name}")
        full_results = self.load_data(model_name, shots, dataset, template, separator, enumerator, choices_order,
                                      max_samples, drop)
        clean_df = self.remove_duplicates(full_results)
        load_time = time.time()
        print(f"The size of the data after removing duplicates is: {len(clean_df)}")
        print(f"Data loading completed in {load_time - start_time:.2f} seconds")
        return clean_df

    def load_data_batched(self, model_name, shots, max_samples):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries.

        Args:
            model_name (str): Name of the model to filter the dataset.
            shots (int): Number of shots to filter the dataset.

        Yields:
            pd.DataFrame: Processed batch of data.
        """
        # Load the dataset from HuggingFace
        if self.dataset is None:
            dataset = load_dataset(self.dataset_name, split=self.split, max_samples=max_samples)
            dataset = dataset.remove_columns(['family', 'generated_text', 'ground_truth'])
            self.dataset = dataset

            # Initial filtering using the `datasets` API
        filtered_dataset = self.dataset.filter(
            lambda x: x['model'] == model_name and x['shots'] == shots
        )

        # Convert to Arrow table for efficient processing
        filtered_flat = filtered_dataset.flatten_indices()
        # filtered_flat.data.table כעת משקף פיזית רק את השורות המסוננות
        arrow_table = filtered_flat.data.table

        # Process in batches
        for batch in arrow_table.to_batches(self.batch_size):
            df_batch = batch.to_pandas()

            # Perform specific processing on each batch
            processed_batch = self.no_process_batch(df_batch)
            yield processed_batch

    def load_data(self, model_name, shots, dataset=None, template=None, separator=None, enumerator=None,
                  choices_order=None, max_samples=None,
                  drop=True):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries.
        """
        # Load the dataset if not already loaded
        if self.dataset is None:
            try:
                if max_samples is not None:
                    split_with_max_samples = f"{self.split}[:{max_samples}]"
                    self.dataset = load_dataset(self.dataset_name, split=split_with_max_samples)
                else:
                    self.dataset = load_dataset(self.dataset_name, split=self.split)
                # print the dataset features (not the data)
                print(self.dataset)
                if drop:
                    self.dataset = self.dataset.remove_columns(['family', 'generated_text', 'ground_truth'])
            except Exception as e:
                raise Exception(f"Error loading dataset: {str(e)}")

        # Get the Arrow table
        print("Extracting Arrow table...")
        arrow_table = self.dataset.data.table

        conditions = [
            pc.equal(arrow_table['model'], model_name),
            pc.equal(arrow_table['shots'], shots)
        ]

        optional_filters = {
            'dataset': dataset,
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

            if max_samples is not None and len(df_filtered) > max_samples:
                df_filtered = df_filtered.head(max_samples)

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

    def process_batch(self, df_batch):
        """
        Processes each batch of data.
        This method should be implemented by subclasses.

        Args:
            df_batch (pd.DataFrame): A batch of the dataset.

        Returns:
            pd.DataFrame: Processed batch of data.
        """
        raise NotImplementedError("The process_batch method should be implemented by subclasses.")

    def no_process_batch(self, df_batch):
        """
        Processes each batch of data.
        """
        # Perform necessary operations on the batch

        # Compute metrics for each batch
        return df_batch

    def process_batch(self, df_batch):
        """
        Processes each batch of data.
        This method should be implemented by subclasses.

        Args:
            df_batch (pd.DataFrame): A batch of the dataset.

        Returns:
            pd.DataFrame: Processed batch of data.
        """
        raise NotImplementedError("The process_batch method should be implemented by subclasses.")

    def process_batch_count_scores(self, df_batch):
        """
        Processes each batch of data.
        """
        # Perform necessary operations on the batch

        # Compute metrics for each batch
        grouped = df_batch.groupby(["dataset", "template", "separator", "enumerator", "choices_order"])
        sum_scores = grouped['score'].sum()
        n_samples = grouped['score'].count()

        processed_df = pd.DataFrame({
            'sum_scores': sum_scores,
            'count': n_samples
        }).reset_index()

        return processed_df
