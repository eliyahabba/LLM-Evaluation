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
            'dimensions_1: enumerator': prompt_config['dimensions_enumerator'],
            'dimensions_2: separator': prompt_config['dimensions_separator'],
            'dimensions_3: choices_order': prompt_config['dimensions_choices_order_method'],
            'dimensions_4: instruction_phrasing_text': prompt_config['dimensions_instruction_phrasing_text'],
            'dimensions_5: shots': prompt_config['dimensions_shots'],
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

            # Extract model and dataset info
            try:
                model_name = file_path.parent.parent.parent.name
                dataset_name = file_path.stem.split('_')[0]  # Remove any suffixes
                self.logger.info(f"Processing model: {model_name}, dataset: {dataset_name}")
            except Exception as e:
                self.logger.error(f"Error extracting model/dataset info from {file_path}: {e}")
                return []

            # Process the data in batches
            try:
                # Create output directory structure
                lang = file_path.parents[1].name
                shots_dir = file_path.parent.name
                output_dir = self.data_dir / model_name / lang / shots_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{file_path.stem}{ProcessingConstants.PARQUET_EXTENSION}"

                # Initialize writer
                writer = None
                processed_rows = 0
                
                # Read and process in batches
                for batch in self._read_parquet_in_batches(file_path):
                    try:
                        # Convert batch to pandas for processing
                        batch_df = batch
                        lean_df = self.create_lean_version(batch_df)
                        
                        if lean_df.empty:
                            continue

                        # Initialize writer with schema from first batch
                        if writer is None:
                            table = pa.Table.from_pandas(lean_df)
                            writer = pq.ParquetWriter(
                                str(output_path),
                                table.schema,
                                version=ParquetConstants.VERSION,
                                write_statistics=ParquetConstants.WRITE_STATISTICS
                            )

                        # Write batch
                        table = pa.Table.from_pandas(lean_df)
                        writer.write_table(table)
                        processed_rows += len(lean_df)
                        self.logger.info(f"Processed {processed_rows} rows so far...")

                    except Exception as e:
                        self.logger.error(f"Error processing batch: {e}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        continue

                # Close writer
                if writer:
                    writer.close()
                    return [str(output_path)]
                return []

            except Exception as e:
                self.logger.error(f"Error processing data for {file_path}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return []

        except Exception as e:
            self.logger.error(f"Unexpected error processing {file_path}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _read_parquet_in_batches(self, file_path: Path):
        """Read a parquet file in batches to conserve memory."""
        try:
            parquet_file = pq.ParquetFile(str(file_path))
            num_row_groups = parquet_file.num_row_groups
            
            # Calculate rows per batch based on total rows and DEFAULT_BATCH_SIZE
            total_rows = sum(parquet_file.metadata.row_group(i).num_rows for i in range(num_row_groups))
            rows_per_batch = min(ProcessingConstants.DEFAULT_BATCH_SIZE, total_rows)
            
            self.logger.info(f"Processing {total_rows:,} rows in batches of {rows_per_batch:,}")
            
            current_row = 0
            current_batch = []
            
            for row_group_idx in range(num_row_groups):
                table = parquet_file.read_row_group(row_group_idx)
                df = table.to_pandas()
                
                for _, row in df.iterrows():
                    current_batch.append(row)
                    current_row += 1
                    
                    if len(current_batch) >= rows_per_batch:
                        yield pd.DataFrame(current_batch)
                        current_batch = []
                        
            # Yield any remaining rows
            if current_batch:
                yield pd.DataFrame(current_batch)
                
        except Exception as e:
            self.logger.error(f"Error reading parquet file in batches: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

if __name__ == "__main__":
    processor = LeanSchemaProcessor(data_dir="/Users/ehabba/Desktop/IBM_Results_Processed/lean_schema")
    file_path = Path("/Users/ehabba/Desktop/IBM_Results_Processed/full_schema/Llama-3.2-1B-Instruct/en/shots_0/mmlu.logical_fallacies.parquet")
    result = processor.process_file(file_path)
