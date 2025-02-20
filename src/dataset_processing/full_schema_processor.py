# data_processor.py
from pathlib import Path
from typing import Dict, List, Optional
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from filelock import FileLock

from base_processor import BaseProcessor
from src.experiments.experiment_preparation.dataset_scheme.conversions.dataset_schema_adapter_ibm import SchemaConverter
from src.experiments.experiment_preparation.dataset_scheme.conversions.RunOutputMerger import RunOutputMerger
from constants import ProcessingConstants, ParquetConstants


class FullSchemaProcessor(BaseProcessor):
    def __init__(self, **kwargs):
        """Initialize the data processor."""
        super().__init__(**kwargs)
        
        # Initialize schema converter
        models_metadata_path = Path(ProcessingConstants.MODELS_METADATA_PATH)
        self.converter = SchemaConverter(models_metadata_path, logger=self.logger)
        
        # Dictionary to store parquet writers
        self.writers = {}

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame using the schema converter more efficiently."""
        try:
            self.logger.info(f"Starting conversion of DataFrame with shape: {df.shape}")
            self.logger.debug(f"DataFrame columns: {df.columns}")
            
            # Convert batch at once instead of row by row
            converted_data = self.converter.convert_dataframe(df, probs=True, logger=self.logger)
            if not converted_data:
                self.logger.warning("No data converted from batch")
                return pd.DataFrame()
            
            # Create DataFrame once
            return pd.DataFrame(converted_data)
        except Exception as e:
            self.logger.error(f"Error processing DataFrame: {str(e)}")
            self.logger.error(f"DataFrame sample: {df.head(1).to_dict()}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def process_file(self, file_path: Path) -> List[str]:
        """
        Process a single file and write results to model/dataset specific files.

        Args:
            file_path: Path to the parquet file

        Returns:
            List of processed file paths
        """
        self.logger.info(f"Processing file: {file_path}")
        processed_files = set()

        try:
            # Initialize RunOutputMerger for this file
            run_merger = RunOutputMerger(file_path, batch_size=self.batch_size)
            num_of_batches = 0
            
            # Process batches using RunOutputMerger
            for batch_df in tqdm(run_merger.process_batches(), desc=f"Processing {file_path.name}"):
                # Process and convert the batch
                converted_df = self.process_dataframe(batch_df)
                
                if converted_df.empty:
                    continue

                # Extract model, dataset and language
                converted_df['model_name'] = converted_df.apply(
                    lambda x: x['model']['model_info']['name'].split("/")[-1], axis=1
                )
                converted_df['dataset_name'] = converted_df.apply(
                    lambda x: x['instance']['sample_identifier']['dataset_name'], axis=1
                )
                converted_df['language'] = converted_df.apply(
                    lambda x: x['instance']['language'], axis=1
                )

                # Group and write each model/dataset/language combination
                for (model, dataset, lang), group_df in converted_df.groupby(['model_name', 'dataset_name', 'language']):
                    # Create directory structure
                    output_dir = self.data_dir / model / lang
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = output_dir / f"{dataset}{ProcessingConstants.PARQUET_EXTENSION}"
                    lock_file = output_dir / f"{dataset}{ProcessingConstants.LOCK_FILE_EXTENSION}"
                    
                    # Remove grouping columns and reset index
                    group_df = group_df.drop(columns=['model_name', 'dataset_name', 'language'])
                    group_df = group_df.reset_index(drop=True)  # Drop the index completely
                    
                    # Convert DataFrame to PyArrow Table
                    table = pa.Table.from_pandas(group_df, preserve_index=False)  # Don't preserve index
                    
                    with FileLock(lock_file):
                        if str(output_path) not in self.writers:
                            self.writers[str(output_path)] = pq.ParquetWriter(
                                str(output_path), 
                                table.schema,
                                version=ParquetConstants.VERSION,
                                write_statistics=ParquetConstants.WRITE_STATISTICS
                            )
                        
                        self.writers[str(output_path)].write_table(table)
                        processed_files.add(str(output_path))
                        self.logger.info(f"Written batch for {model}/{lang}/{dataset}")

                num_of_batches += 1
                if ProcessingConstants.DEFAULT_NUM_BATCHES is not None and num_of_batches == ProcessingConstants.DEFAULT_NUM_BATCHES:
                    break

            # Close all writers
            for key in list(self.writers.keys()):
                self.writers[key].close()
                del self.writers[key]

            return list(processed_files)

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            # Clean up writers on error
            for writer in self.writers.values():
                try:
                    writer.close()
                except:
                    pass
            self.writers.clear()
            return []

    def __del__(self):
        """Cleanup writers on object destruction."""
        for writer in self.writers.values():
            try:
                writer.close()
            except:
                pass