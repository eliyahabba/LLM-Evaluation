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
        """Process a single file and write results to model/dataset specific files."""
        self.logger.info(f"Processing file: {file_path}")
        processed_files = set()

        try:
            # Initialize RunOutputMerger for this file
            run_merger = RunOutputMerger(file_path, batch_size=self.batch_size)
            num_of_batches = 0

            # Create schema once
            schema = self._get_schema()

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
                    group_df = group_df.reset_index(drop=True)

                    # Convert DataFrame to PyArrow Table using predefined schema
                    table = pa.Table.from_pandas(group_df, schema=schema)

                    with FileLock(lock_file):
                        if str(output_path) not in self.writers:
                            self.writers[str(output_path)] = pq.ParquetWriter(
                                str(output_path),
                                schema,  # Use the predefined schema
                                version=ParquetConstants.VERSION,
                                write_statistics=ParquetConstants.WRITE_STATISTICS
                            )

                        self.writers[str(output_path)].write_table(table)
                        processed_files.add(str(output_path))
                        # self.logger.info(f"Written batch for {model}/{lang}/{dataset}")

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

    def __del__(self):
        """Cleanup writers on object destruction."""
        for writer in self.writers.values():
            try:
                writer.close()
            except:
                pass