from typing import Dict, Optional, Any, Iterator

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset


class RunOutputMerger:
    """A class to process and merge run data with associated metadata."""

    def __init__(self, parquet_path: str, batch_size: int = 1000):
        """
        Initialize the RunDataProcessor.

        Args:
            parquet_path (str): Path to the parquet file
            batch_size (int): Size of batches for processing
        """
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.runs_dict = self._load_runs_dataset()

    def _load_runs_dataset(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and process the runs dataset into a dictionary for O(1) lookups.

        Returns:
            Dict[str, Dict[str, Any]]: Processed runs dictionary
        """
        runs_ds = load_dataset("OfirArviv/HujiCollabRun")
        return {
            row['run_id']: {k: v for k, v in row.items() if k != 'run_id'}
            for row in runs_ds['run']
        }

    def _get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run information for a specific run_id.

        Args:
            run_id (str): The run ID to look up

        Returns:
            Optional[Dict[str, Any]]: Run information if found, None otherwise
        """
        return self.runs_dict.get(run_id)

    def _add_run_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add run-related columns to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with added run columns
        """
        # Map run information to DataFrame
        run_info = df['run_id'].map(self._get_run_info)
        # Add all remaining columns with 'run_' prefix
        run_columns = list(next(iter(self.runs_dict.values())).keys())
        for col in run_columns:
            df[f'run_{col}'] = run_info.apply(
                lambda x: x[col] if x is not None else None
            )

        return df

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
    parquet_path = "~/Downloads/data_sample.parquet"
    processor = RunOutputMerger(parquet_path)
    for processed_batch in processor.process_batches():
        # Now you can work with each batch
        print(f"Batch shape: {processed_batch.shape}")
        print(processed_batch.head())
        # Add any additional batch processing here
        break

if __name__ == "__main__":
    main()
