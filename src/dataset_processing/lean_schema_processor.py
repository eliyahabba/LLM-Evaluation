from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from base_processor import BaseProcessor
from constants import ParquetConstants, ProcessingConstants


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
        self.logger.info(f"Processing file: {file_path}")
        processed_files = set()

        try:
            # Read the full schema file
            df = pd.read_parquet(file_path)
            if df.empty:
                return []

            # Convert to lean schema
            lean_df = self.create_lean_version(df)
            if lean_df.empty:
                return []

            # Get model/lang/shots/dataset from the file path
            # Path structure is: full_schema_dir/model/lang/shots/dataset.parquet
            model = file_path.parent.parent.parent.parent.name
            lang = file_path.parent.parent.parent.name
            shots_dir = file_path.parent.parent.name  # e.g. "3_shot"
            dataset = file_path.parent.name

            # Create directory structure - maintain same structure as full schema
            output_dir = self.data_dir / model / lang / shots_dir / dataset
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{file_path.stem}{ProcessingConstants.PARQUET_EXTENSION}"

            # Convert DataFrame to PyArrow Table
            schema = lean_df.to_parquet().schema
            table = pa.Table.from_pandas(lean_df, schema)

            # Write to unique file for this process
            if str(output_path) not in self.writers:
                self.writers[str(output_path)] = pq.ParquetWriter(
                    str(output_path),
                    schema,
                    version=ParquetConstants.VERSION,
                    write_statistics=ParquetConstants.WRITE_STATISTICS
                )

            self.writers[str(output_path)].write_table(table)
            processed_files.add(str(output_path))

            # Close writer
            if str(output_path) in self.writers:
                self.writers[str(output_path)].close()
                del self.writers[str(output_path)]

            return list(processed_files)

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return []
