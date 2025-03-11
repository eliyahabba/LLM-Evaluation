from pathlib import Path
from typing import Dict, Optional, Iterator

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset


class RunOutputMerger:
    """A class to process and merge run data with associated metadata."""

    def __init__(self, parquet_path: Path, batch_size: int = 1000):
        """
        Initialize the RunDataProcessor.

        Args:
            parquet_path (str): Path to the parquet file
            batch_size (int): Size of batches for processing
        """
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.runs_df = self._load_runs_dataset()

    def _load_runs_dataset(self) -> pd.DataFrame:
        """
        Load and process the runs dataset into a dictionary for O(1) lookups.

        Returns:
            Dict[str, Dict[str, Any]]: Processed runs dictionary
        """
        runs_ds = load_dataset("OfirArviv/HujiCollabRun")
        train_runs_ds = runs_ds['run']
        df = train_runs_ds.to_pandas()
        return df

    def _add_run_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add run-related columns to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with added run columns
        """

        # Map run information to DataFrame
        run_df_unique = self.runs_df.drop_duplicates(subset=['run_id'])
        run_df_unique.columns = ['run_' + col if col != 'run_id' else col for col in run_df_unique.columns]

        merged_df = df.merge(
            run_df_unique,
            on='run_id',
            how='left'
        )
        merged_df.dropna(subset=['run_model_args'], inplace=True)
        return merged_df

    def process_batches(self) -> Iterator[pd.DataFrame]:
        """
        Process the parquet file in batches, adding run information.

        Yields:
            pd.DataFrame: Processed batch with added run information
        """
        try:
            parquet_file = pq.ParquetFile(self.parquet_path)

            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                df = self._add_run_columns(df)
                yield df

        except Exception as e:
            print(f"Error processing parquet file: {str(e)}")
            raise


def main():
    """Main function to demonstrate usage."""
    f = "data_2025-02-10.parquet"
    parquet_path = Path(f"~/Downloads/{f}")
    processor = RunOutputMerger(parquet_path)
    num_of_batches = 0
    for processed_batch in processor.process_batches():
        # Now you can work with each batch
        # print(f"Batch shape: {processed_batch.shape}")
        num_of_batches += 1
        print(f"Batches processed: {num_of_batches}")
        print(processed_batch.head())
        # # Add any additional batch processing here
        break
    print(f"Processed {num_of_batches} batches.")


if __name__ == "__main__":
    main()
