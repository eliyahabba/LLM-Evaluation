# data_loader.py
import time
from typing import Dict, List, Any
from typing import Optional

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
        self.load_data(max_samples, drop)
        full_results = self.extract_data(model_name, shots,
                                         dataset=dataset, template=template, separator=separator, enumerator=enumerator,
                                         choices_order=choices_order)
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
        arrow_table = filtered_flat.data.table

        # Process in batches
        for batch in arrow_table.to_batches(self.batch_size):
            df_batch = batch.to_pandas()

            # Perform specific processing on each batch
            processed_batch = self.no_process_batch(df_batch)
            yield processed_batch

    def load_data(self, max_samples=None,
                  drop=True):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries.
        """
        # Load the dataset if not already loaded

        num_proc = 8
        if self.dataset is None:
            try:
                if max_samples is not None:
                    split_with_max_samples = f"{self.split}[:{max_samples}]"
                    self.dataset = load_dataset(self.dataset_name, split=split_with_max_samples, num_proc=num_proc)
                else:
                    self.dataset = load_dataset(self.dataset_name, split=self.split, num_proc=num_proc)
                # print the dataset features (not the data)
                print(self.dataset)
                if drop:
                    self.dataset = self.dataset.remove_columns(['family', 'generated_text', 'ground_truth'])
            except Exception as e:
                raise Exception(f"Error loading dataset: {str(e)}")

    def extract_data2(
            self,
            model_name: str,
            shots: int,
            dataset: Optional[str] = None,
            template: Optional[str] = None,
            separator: Optional[str] = None,
            enumerator: Optional[str] = None,
            choices_order: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Efficiently filter a large dataset using batched processing and parallel execution.

        Args:
            model_name: Name of the model to filter by
            shots: Number of shots to filter by
            dataset: Optional dataset name filter
            template: Optional template filter
            separator: Optional separator filter
            enumerator: Optional enumerator filter
            choices_order: Optional choices order filter

        Returns:
            pd.DataFrame: Filtered dataset as a pandas DataFrame
        """
        try:
            print("Starting data filtering process...")

            # Step 1: Define the columns we need to keep
            # Add any additional columns you need in the final output
            required_columns = [
                'model',
                'shots',
                'dataset',
                'template',
                'separator',
                'enumerator',
                'choices_order'
            ]

            # Step 2: Select only necessary columns to reduce memory usage
            dataset_subset = self.dataset.select_columns(required_columns)

            # Step 3: Define the batch filtering function
            def filter_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[bool]]:
                """
                Process a batch of examples and return a boolean mask for filtering.

                Args:
                    examples: Dictionary of column names to lists of values

                Returns:
                    Dictionary with a single 'keep' key containing boolean mask
                """
                batch_size = len(examples['model'])
                # Initialize all rows as True
                mask = [True] * batch_size

                # Apply each filter condition if it exists
                # Required filters
                mask = [m and (mod == model_name) for m, mod in zip(mask, examples['model'])]
                mask = [m and (s == shots) for m, s in zip(mask, examples['shots'])]

                # Optional filters
                if dataset is not None:
                    mask = [m and (d == dataset) for m, d in zip(mask, examples['dataset'])]

                if template is not None:
                    mask = [m and (t == template) for m, t in zip(mask, examples['template'])]

                if separator is not None:
                    mask = [m and (sep == separator) for m, sep in zip(mask, examples['separator'])]

                if enumerator is not None:
                    mask = [m and (e == enumerator) for m, e in zip(mask, examples['enumerator'])]

                if choices_order is not None:
                    mask = [m and (c == choices_order) for m, c in zip(mask, examples['choices_order'])]

                return {'keep': mask}

            # Step 4: Apply the batched filtering
            print("Applying filters...")
            filtered_dataset = dataset_subset.map(
                filter_batch,
                batched=True,
                batch_size=100000,  # Adjust based on available memory
                num_proc=4,
                remove_columns=dataset_subset.column_names,
                load_from_cache_file=True,
                desc="Filtering dataset"
            )

            # Step 5: Convert to pandas DataFrame
            print("Converting to pandas DataFrame...")
            df_filtered = filtered_dataset.to_pandas()

            # Step 6: Print summary statistics
            print("\nFiltering results:")
            print(f"Total rows after filtering: {len(df_filtered)}")
            print("\nModel distribution:")
            print(df_filtered['model'].value_counts())

            return df_filtered

        except Exception as e:
            raise Exception(f"Error during data filtering: {str(e)}")

    def extract_data(self, model_name, shots, dataset=None, template=None, separator=None, enumerator=None,
                     choices_order=None):

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
