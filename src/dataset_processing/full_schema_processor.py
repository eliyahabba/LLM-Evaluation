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
            # self.logger.info(f"Starting conversion of DataFrame with shape: {df.shape}")
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
        """Process a single file and write results to model/dataset specific files."""
        self.logger.info(f"Processing file: {file_path}")
        processed_files = set()
        created_files = set()  # Track all files created during processing

        try:
            # Initialize RunOutputMerger for this file
            run_merger = RunOutputMerger(file_path, batch_size=self.batch_size)
            num_of_batches = 0

            # Create schema once
            schema = self._get_schema()

            # Process batches using RunOutputMerger
            for batch_df in tqdm(run_merger.process_batches(), desc=f"Processing {file_path.name}"):
                try:
                    # Process and convert the batch
                    converted_df = self.process_dataframe(batch_df)

                    if converted_df.empty:
                        continue

                    # Extract model, dataset, language and shots
                    converted_df['model_name'] = converted_df.apply(
                        lambda x: x['model']['model_info']['name'].split("/")[-1], axis=1
                    )
                    converted_df['dataset_name'] = converted_df.apply(
                        lambda x: x['instance']['sample_identifier']['dataset_name'], axis=1
                    )
                    converted_df['language'] = converted_df.apply(
                        lambda x: x['instance']['language'], axis=1
                    )
                    converted_df['shots'] = converted_df.apply(
                        lambda x: x['prompt_config']['dimensions']['shots'], axis=1
                    )

                    # Group and write each model/dataset/language/shots combination
                    for (model, dataset, lang, shots), group_df in converted_df.groupby(['model_name', 'dataset_name', 'language', 'shots']):
                        # Create shots directory name
                        shots_dir = f"{shots}_shot"
                        # Create directory structure with shots
                        output_dir = self.data_dir / model / lang / shots_dir / dataset
                        output_dir.mkdir(parents=True, exist_ok=True)

                        output_path = output_dir / f"{file_path.stem}{ProcessingConstants.PARQUET_EXTENSION}"
                        created_files.add(output_path)  # Track the file

                        # Remove grouping columns and reset index
                        group_df = group_df.drop(columns=['model_name', 'dataset_name', 'language', 'shots'])
                        group_df = group_df.reset_index(drop=True)

                        # Convert DataFrame to PyArrow Table using predefined schema
                        table = pa.Table.from_pandas(group_df, schema=schema)

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

                    num_of_batches += 1
                    if ProcessingConstants.DEFAULT_NUM_BATCHES is not None and num_of_batches == ProcessingConstants.DEFAULT_NUM_BATCHES:
                        break

                except Exception as batch_e:
                    self.logger.error(f"Error processing batch in {file_path}: {batch_e}")
                    import traceback
                    self.logger.error(f"Batch error traceback: {traceback.format_exc()}")
                    # Continue with next batch

            # Close all writers properly
            self._close_writers()

            if processed_files:
                return list(processed_files)
            else:
                # If no files were successfully processed, clean up any partially created files
                self._cleanup_failed_files(created_files)
                return []

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Clean up on error
            self._close_writers()
            self._cleanup_failed_files(created_files)
            return []

    def _get_schema(self):
        """Create the full schema for parquet files."""
        return pa.schema([
            ('evaluation_id', pa.string()),
            ('model', pa.struct({
                'configuration': pa.struct({
                    'architecture': pa.string(),
                    'context_window': pa.int64(),
                    'hf_path': pa.string(),
                    'is_instruct': pa.bool_(),
                    'parameters': pa.int64(),
                    'revision': pa.null()
                }),
                'inference_settings': pa.struct({
                    'generation_args': pa.struct({
                        'max_tokens': pa.int64(),
                        'stop_sequences': pa.list_(pa.null()),
                        'temperature': pa.null(),
                        'top_k': pa.int64(),
                        'top_p': pa.null(),
                        'use_vllm': pa.bool_()
                    }),
                    'quantization': pa.struct({
                        'bit_precision': pa.string(),
                        'method': pa.string()
                    })
                }),
                'model_info': pa.struct({
                    'family': pa.string(),
                    'name': pa.string()
                })
            })),
            ('prompt_config', pa.struct({
                'dimensions': pa.struct({
                    'choices_order': pa.struct({
                        'description': pa.string(),
                        'method': pa.string()
                    }),
                    'demonstrations': pa.list_(pa.struct({
                        'topic': pa.string(),
                        'question': pa.string(),
                        'choices': pa.list_(pa.struct({
                            'id': pa.string(),
                            'text': pa.string()
                        })),
                        'answer': pa.int64()
                    })),
                    'enumerator': pa.string(),
                    'instruction_phrasing': pa.struct({
                        'name': pa.string(),
                        'text': pa.string()
                    }),
                    'separator': pa.string(),
                    'shots': pa.int64()
                }),
                'prompt_class': pa.string()
            })),
            ('instance', pa.struct({
                'classification_fields': pa.struct({
                    'choices': pa.list_(pa.struct({
                        'id': pa.string(),
                        'text': pa.string()
                    })),
                    'ground_truth': pa.struct({
                        'id': pa.string(),
                        'text': pa.string()
                    }),
                    'question': pa.string()
                }),
                'language': pa.string(),
                'perplexity': pa.float64(),
                'prompt_logprobs': pa.list_(pa.list_(pa.struct({
                    'decoded_token': pa.string(),
                    'logprob': pa.float64(),
                    'rank': pa.int64(),
                    'token_id': pa.int64()
                }))),
                'raw_input': pa.string(),
                'sample_identifier': pa.struct({
                    'dataset_name': pa.string(),
                    'hf_index': pa.int64(),
                    'hf_repo': pa.string(),
                    'hf_split': pa.string()
                }),
                'task_type': pa.string()
            })),
            ('output', pa.struct({
                'response': pa.string(),
                'cumulative_logprob': pa.float64(),
                'generated_tokens_logprobs': pa.list_(pa.list_(pa.struct({
                    'decoded_token': pa.string(),
                    'logprob': pa.float64(),
                    'rank': pa.int64(),
                    'token_id': pa.int64()
                })))
            })),
            ('evaluation', pa.struct({
                'ground_truth': pa.string(),
                'evaluation_method': pa.struct({
                    'method_name': pa.string(),
                    'description': pa.string(),
                    'closest_answer': pa.string()
                }),
                'score': pa.float64()
            }))
        ])

    def _close_writers(self):
        """Safely close all writers."""
        for key in list(self.writers.keys()):
            try:
                self.writers[key].close()
            except Exception as e:
                self.logger.error(f"Error closing writer for {key}: {e}")
            finally:
                del self.writers[key]

    def _cleanup_failed_files(self, file_paths: set):
        """Clean up files created during failed processing."""
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.info(f"Cleaned up failed file: {file_path}")
            except Exception as e:
                self.logger.error(f"Error cleaning up file {file_path}: {e}")

    def __del__(self):
        """Ensure writers are closed on object destruction."""
        self._close_writers()