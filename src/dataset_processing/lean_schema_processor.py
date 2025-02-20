from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock
from base_processor import BaseProcessor
from constants import SchemaConstants, ParquetConstants, ProcessingConstants


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
        """Process a full schema file into lean schema."""
        try:
            # Read full schema file
            df = pd.read_parquet(file_path)
            
            # Create lean version
            lean_df = self.create_lean_version(df)
            
            # Get model/lang/dataset from the file path
            # Path structure is: full_schema_dir/model/lang/dataset.parquet
            model = file_path.parent.parent.name
            lang = file_path.parent.name
            dataset = file_path.stem
            
            # Create output directory structure
            output_dir = self.data_dir / model / lang
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{dataset}{ProcessingConstants.PARQUET_EXTENSION}"
            
            # Write file with locking
            lock_file = output_dir / f"{dataset}{ProcessingConstants.LOCK_FILE_EXTENSION}"
            
            with FileLock(lock_file):
                table = pa.Table.from_pandas(lean_df)
                pq.write_table(
                    table, 
                    output_path,
                    version=ParquetConstants.VERSION,
                    write_statistics=ParquetConstants.WRITE_STATISTICS
                )
            
            return [str(output_path)]

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return [] 