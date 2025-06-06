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
        # first need to eval each column to convert from string to dict
        for col in df.columns:
            if col in ['evaluation_id']:
                continue
            try:
                import json
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            except Exception as e:
                self.logger.error(f"Error evaluating column {col}: {e}")
                continue

        instance = pd.json_normalize(df['instance'].tolist())
        model = pd.json_normalize(df['model'].tolist())
        prompt_config = pd.json_normalize(df['prompt_config'].tolist(), sep='_')
        output = pd.json_normalize(df['output'].tolist())
        evaluation = pd.json_normalize(df['evaluation'].tolist())
        dimensions_2 = prompt_config['dimensions_separator']
        # repace the separator from "\\n" to "\n" (only if this the case)
        dimensions_2 = dimensions_2.apply(lambda x: x.replace("\\n", "\n") if isinstance(x, str) else x)
        # Create lean DataFrame directly
        return pd.DataFrame({
            'evaluation_id': df['evaluation_id'],
            'dataset': instance['sample_identifier.dataset_name'],
            'sample_index': instance['sample_identifier.hf_index'],
            'model': model['model_info.name'],
            'quantization': model['inference_settings.quantization.bit_precision'],
            'dimensions_1: enumerator': prompt_config['dimensions_enumerator'],
            'dimensions_2: separator': dimensions_2,
            'dimensions_3: choices_order': prompt_config['dimensions_choices_order_method'],
            'dimensions_4: instruction_phrasing_text': prompt_config['dimensions_instruction_phrasing_text'],
            'dimensions_5: shots': prompt_config['dimensions_shots'],
            'raw_input': instance['raw_input'],
            'generated_text': output['response'],
            'cumulative_logprob': output['cumulative_logprob'],
            'closest_answer': output['response'],
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

                # Initialize writer and counters
                writer = None
                processed_rows = 0

                # Count total rows in input file
                parquet_file = pq.ParquetFile(str(file_path))
                total_input_rows = sum(parquet_file.metadata.row_group(i).num_rows
                                     for i in range(parquet_file.num_row_groups))

                # Read and process in batches
                for batch in self._read_parquet_in_batches(file_path):
                    try:
                        # Convert batch to pandas for processing
                        batch = batch.reset_index(drop=True)
                        lean_df = self.create_lean_version(batch)

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

                    except Exception as e:
                        self.logger.error(f"Error processing batch: {e}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        continue

                # Close writer and log statistics
                if writer:
                    writer.close()

                    # Log row count differences
                    rows_diff = total_input_rows - processed_rows
                    if rows_diff != 0:
                        self.logger.warning(
                            f"Row count mismatch in {file_path.name}:\n"
                            f"  - Original rows: {total_input_rows:,}\n"
                            f"  - Lean schema rows: {processed_rows:,}\n"
                            f"  - Missing/extra rows: {abs(rows_diff):,}\n"
                            f"  - Difference percentage: {(abs(rows_diff) / total_input_rows * 100):.2f}%"
                        )

                    self.logger.info(
                        f"Completed processing {file_path.name}:\n"
                        f"  - Original rows: {total_input_rows:,}\n"
                        f"  - Lean schema rows: {processed_rows:,}\n"
                        f"  - Reduction: {((total_input_rows - processed_rows) / total_input_rows * 100):.2f}%"
                    )
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

    from pathlib import Path
    import glob
    import os
    from tqdm import tqdm


    def process_all_files(files_dir, results_dir, processor):
        # Get all models (directories at the first level)
        all_models = [d for d in os.listdir(files_dir) if os.path.isdir(os.path.join(files_dir, d))]

        results = []

        # Create a list of all files to process first, so we can show a proper progress bar
        all_file_paths = []

        for model in all_models:
            model_dir = os.path.join(files_dir, model)

            # Get all languages (directories within the model directory)
            all_langs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

            for lang in all_langs:
                lang_dir = os.path.join(model_dir, lang)

                # Get all shot directories
                all_shots = [d for d in os.listdir(lang_dir) if
                             os.path.isdir(os.path.join(lang_dir, d)) and d.startswith("shots_")]

                for shot in all_shots:
                    shot_dir = os.path.join(lang_dir, shot)

                    # Find ALL parquet files in this directory
                    parquet_files = glob.glob(os.path.join(shot_dir, "*.parquet"))

                    for file_path in parquet_files:
                        all_file_paths.append({
                            'file_path': file_path,
                            'model': model,
                            'language': lang,
                            'shots': shot,
                            'file': os.path.basename(file_path)
                        })

        print(f"Found {len(all_file_paths)} parquet files to process")

        # Now process all files with a progress bar
        for file_info in tqdm(all_file_paths, desc="Processing files"):
            file_path = file_info['file_path']
            try:
                result = processor.process_file(Path(file_path))
                file_info['result'] = result
                results.append(file_info)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        return results


    # Usage
    files_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/converted_data_paerqut"
    results_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/converted_data_paerqut_lite_version"
    processor = LeanSchemaProcessor(data_dir=results_dir)
    all_results = process_all_files(files_dir, results_dir, processor)

    # You can then analyze or save the results
    print(f"Successfully processed {len(all_results)} files")

    # processor = LeanSchemaProcessor(data_dir="/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/converted_data_paerqut_lite_version")
    # file_path = Path("/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/converted_data_paerqut/01-ai_yi-6b/en/shots_5/mmlu.abstract_algebra.parquet")
    # result = processor.process_file(file_path)
