from pathlib import Path
from typing import List
import os
import traceback

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from base_processor import BaseProcessor
from constants import ParquetConstants, ProcessingConstants
from src.dataset_processing.logger_config import LoggerConfig


class LeanSchemaProcessor(BaseProcessor):
    def __init__(self, **kwargs):
        """Initialize the lean schema processor."""
        super().__init__(**kwargs)
        self.writers = {}

    def create_lean_version(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert full schema DataFrame to lean version more efficiently."""
        # Extract nested structures once
        instance = pd.json_normalize(df['instance'].tolist())
        model = pd.json_normalize(df['model'].tolist())
        prompt_config = pd.json_normalize(df['prompt_config'].tolist(), sep='_')
        output = pd.json_normalize(df['output'].tolist())
        evaluation = pd.json_normalize(df['evaluation'].tolist())

        # Create lean DataFrame directly
        return pd.DataFrame({
            'evaluation_id': df['evaluation_id'],
            'dataset': instance['sample_identifier.dataset_name'],
            'sample_index': instance['sample_identifier.hf_index'],
            'model': model['model_info.name'],
            'quantization': model['inference_settings.quantization.bit_precision'],
            'shots': prompt_config['dimensions_shots'],
            'instruction_phrasing_text': prompt_config['dimensions_instruction_phrasing_text'],
            'instruction_phrasing_name': prompt_config['dimensions_instruction_phrasing_name'],
            'separator': prompt_config['dimensions_separator'],
            'enumerator': prompt_config['dimensions_enumerator'],
            'choices_order': prompt_config['dimensions_choices_order_method'],
            'raw_input': instance['raw_input'],
            'generated_text': output['response'],
            'cumulative_logprob': output['cumulative_logprob'],
            'closest_answer': evaluation['evaluation_method.closest_answer'],
            'ground_truth': evaluation['ground_truth'],
            'score': evaluation['score']
        })

    def process_file(self, file_path: Path) -> List[str]:
        """Process a single file and write results to model/dataset specific files."""
        try:
            self.logger.info(f"Processing {file_path} for lean schema")
            
            # Create logger for this process
            if not hasattr(self, 'logger'):
                self.logger = LoggerConfig.setup_logger(
                    "LeanSchemaProcessor",
                    Path(self.data_dir) / ProcessingConstants.LOGS_DIR_NAME,
                    process_id=os.getpid()
                )

            # Read the parquet file
            try:
                df = pl.read_parquet(str(file_path))
                self.logger.info(f"Successfully read {file_path} with {len(df):,} rows")
            except Exception as e:
                self.logger.error(f"Error reading parquet file {file_path}: {e}")
                return []

            # Extract model and dataset info
            try:
                model_name = file_path.parent.parent.parent.name
                dataset_name = file_path.stem.split('_')[0]  # Remove any suffixes
                self.logger.info(f"Processing model: {model_name}, dataset: {dataset_name}")
            except Exception as e:
                self.logger.error(f"Error extracting model/dataset info from {file_path}: {e}")
                return []

            # Process the data
            try:
                # Convert to lean schema
                lean_df = self.create_lean_version(df.to_pandas())
                if lean_df.empty:
                    return []

                # Get model/lang/shots/dataset from the file path
                lang = file_path.parent.parent.parent.name
                shots_dir = file_path.parent.parent.name  # e.g. "3_shot"

                # Create directory structure - maintain same structure as full schema
                output_dir = self.data_dir / model_name / lang / shots_dir / dataset_name
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / f"{file_path.stem}{ProcessingConstants.PARQUET_EXTENSION}"

                # Convert DataFrame to PyArrow Table directly
                table = pa.Table.from_pandas(lean_df)

                # Write to unique file for this process
                if str(output_path) not in self.writers:
                    self.writers[str(output_path)] = pq.ParquetWriter(
                        str(output_path),
                        table.schema,  # Use schema from the table
                        version=ParquetConstants.VERSION,
                        write_statistics=ParquetConstants.WRITE_STATISTICS
                    )

                self.writers[str(output_path)].write_table(table)
                processed_files = [str(output_path)]

                # Close writer
                if str(output_path) in self.writers:
                    self.writers[str(output_path)].close()
                    del self.writers[str(output_path)]

                self.logger.info(f"Successfully processed data for {model_name}/{dataset_name}")
                return processed_files

            except Exception as e:
                self.logger.error(f"Error processing data for {file_path}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return []

        except Exception as e:
            self.logger.error(f"Unexpected error processing {file_path}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
