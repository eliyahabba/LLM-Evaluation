import concurrent
import logging
import os
import shutil
import uuid
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Set

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


class IncrementalDatasetSplitter:
    def __init__(self, input_dir: str, output_dir: str, num_workers: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"
        self.num_workers = num_workers
        self.processed_files_path = self.output_dir / "processed_files.json"

        self.setup_logger()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Load previously processed files
        self.processed_files = self.load_processed_files()

        self.logger.info(f"Initialized splitter with {num_workers} workers")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Found {len(self.processed_files)} previously processed files")

    def load_processed_files(self) -> Set[str]:
        if self.processed_files_path.exists():
            try:
                with open(self.processed_files_path, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                self.logger.error(f"Error loading processed files list: {e}")
                return set()
        return set()

    def save_processed_files(self):
        try:
            with open(self.processed_files_path, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            self.logger.error(f"Error saving processed files list: {e}")

    def get_new_files(self) -> List[Path]:
        all_files = set(str(f) for f in self.input_dir.glob("*.parquet"))
        new_files = all_files - self.processed_files
        self.logger.info(f"Found {len(new_files)} new files to process")
        # take only the 10 files for testing
        new_files = list(new_files)[:10]
        return [Path(f) for f in new_files]

    def setup_logger(self):
        self.logger = logging.getLogger('DatasetSplitter')
        self.logger.setLevel(logging.INFO)

        log_filename = f"split_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name by replacing slashes with underscores."""
        return model_name.split('/')[-1]

    def get_hierarchical_path(self, model: str, shots: int, dataset: str) -> Path:
        """Create hierarchical path structure for output files."""
        # Sanitize the model name before creating the path
        safe_model = self.sanitize_model_name(model)
        # Create the path with the full hierarchy
        full_path = self.output_dir / safe_model / f"shots_{shots}" / "english" / f"{dataset}.parquet"
        return full_path

    def process_single_file(self, input_file: Path) -> List[str]:
        worker_id = uuid.uuid4()
        temp_files = set()

        self.logger.info(f"Starting to process file: {input_file}")
        try:
            parquet_file = pq.ParquetFile(input_file)
            total_rows = parquet_file.metadata.num_rows

            temp_dfs = {}

            with tqdm(total=total_rows, desc=f"Processing {input_file.name}", unit="rows") as pbar:
                for batch in parquet_file.iter_batches(batch_size=10000):
                    df = batch.to_pandas()
                    grouped = df.groupby(['model', 'shots', 'dataset'])

                    for (model, shots, dataset), group_df in grouped:
                        safe_model = self.sanitize_model_name(model)

                        key = f"{worker_id}_{safe_model}_shots{shots}_{dataset}"
                        if key in temp_dfs:
                            temp_dfs[key] = pd.concat([temp_dfs[key], group_df], ignore_index=True)
                        else:
                            temp_dfs[key] = group_df

                    pbar.update(len(df))

            for key, df in temp_dfs.items():
                temp_file = self.temp_dir / f"{key}.parquet"
                os.makedirs(temp_file.parent, exist_ok=True)
                df.to_parquet(temp_file, index=False)
                temp_files.add(str(temp_file))
                self.logger.info(f"Wrote {len(df)} rows to {temp_file}")

        except Exception as e:
            self.logger.error(f"Fatal error processing file {input_file}: {str(e)}", exc_info=True)
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up temp file {temp_file}: {str(cleanup_error)}")
            return []

        return list(temp_files)

    def merge_temp_files(self):
        """Merge temporary files into the final hierarchical structure."""
        self.logger.info("Starting to merge temporary files")

        triplet_files_map = {}

        # Collect all temporary files
        for temp_file in self.temp_dir.glob("*.parquet"):
            try:
                triplet_key = temp_file.stem
                if triplet_key not in triplet_files_map:
                    triplet_files_map[triplet_key] = []
                triplet_files_map[triplet_key].append(temp_file)
            except Exception as e:
                self.logger.error(f"Error processing temp file {temp_file}: {str(e)}")

        self.logger.info(f"Found {len(triplet_files_map)} unique triplets to merge")

        with tqdm(total=len(triplet_files_map), desc="Merging files by triplet") as pbar:
            for triplet_key, temp_files in triplet_files_map.items():
                try:
                    self.logger.info(f"Processing triplet {triplet_key} with {len(temp_files)} files")

                    # Parse the triplet key to get model, shots, and dataset
                    # Format is: UUID_model_shotsN_dataset
                    key_parts = triplet_key.split('_')

                    # Find the shots part
                    shots_index = -1
                    for i, part in enumerate(key_parts):
                        if part.startswith('shots'):
                            shots_index = i
                            break

                    if shots_index == -1:
                        raise ValueError(f"Could not find 'shots' in key: {triplet_key}")

                    # Extract parts
                    model = '_'.join(key_parts[1:shots_index])  # Everything between UUID and shots is model
                    shots = int(key_parts[shots_index][5:])  # Get number after 'shots'
                    dataset = '_'.join(key_parts[shots_index + 1:])  # Rest is dataset

                    # Create hierarchical path and ensure parent directory exists
                    output_file = self.get_hierarchical_path(model, shots, dataset)
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Load existing data if any
                    dfs = []
                    if output_file.exists():
                        try:
                            existing_df = pd.read_parquet(output_file)
                            dfs.append(existing_df)
                            self.logger.info(f"Loaded existing file {output_file} with {len(existing_df)} rows")
                        except Exception as e:
                            self.logger.error(f"Error reading existing file {output_file}: {str(e)}")

                    # Add new data
                    for temp_file in temp_files:
                        try:
                            df = pd.read_parquet(temp_file)
                            dfs.append(df)
                        except Exception as e:
                            self.logger.error(f"Error reading temp file {temp_file}: {str(e)}")

                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=True)
                        combined_df.to_parquet(output_file, index=False)
                        self.logger.info(f"Successfully created/updated {output_file} with {len(combined_df)} rows")

                except Exception as e:
                    self.logger.error(f"Error processing triplet {triplet_key}: {str(e)}", exc_info=True)

                pbar.update(1)

    def check_single_file_duplicates(self, file_path: Path) -> tuple[bool, str]:
        """Check and clean duplicates in a single file."""
        try:
            df = pd.read_parquet(file_path)
            if 'evaluation_id' not in df.columns:
                return True, f"File {file_path} does not contain evaluation_id column"

            duplicate_count = df.evaluation_id.duplicated().sum()
            if duplicate_count > 0:
                df_cleaned = df.drop_duplicates(subset=['evaluation_id'], keep='first')
                removed_count = len(df) - len(df_cleaned)
                df_cleaned.to_parquet(file_path, index=False)
                return True, f"Cleaned {file_path}: removed {removed_count} duplicate rows"
            else:
                return True, f"No duplicates found in {file_path}"

        except Exception as e:
            return False, f"Error processing file {file_path}: {str(e)}"

    def check_and_clean_duplicates(self):
        """Check all output files for duplicate evaluation_ids and clean them in parallel."""
        self.logger.info("Starting parallel duplicate check and cleanup process...")

        # Collect all parquet files
        parquet_files = []
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(Path(root) / file)

        self.logger.info(f"Found {len(parquet_files)} parquet files to check")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for file_path in parquet_files:
                futures.append(executor.submit(self.check_single_file_duplicates, file_path))

            success_count = 0
            with tqdm(total=len(futures), desc="Checking for duplicates") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    success, message = future.result()
                    if success:
                        success_count += 1
                        if "removed" in message:
                            self.logger.warning(message)
                        else:
                            self.logger.info(message)
                    else:
                        self.logger.error(message)
                    pbar.update(1)

        self.logger.info(f"Completed duplicate check on {success_count} out of {len(parquet_files)} files")

    def process_all_files(self):
        """Process only new files that haven't been processed before."""
        self.logger.info("Starting processing new files")
        start_time = datetime.now()

        new_files = self.get_new_files()
        self.logger.info(f"Found {len(new_files)} new parquet files to process")

        if not new_files:
            self.logger.info("No new files to process")
            return []

        failed_files = []
        processed_files = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {executor.submit(self.process_single_file, f): f for f in new_files}

            with tqdm(total=len(new_files), desc="Processing files", unit="file") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    input_file = future_to_file[future]
                    try:
                        temp_files = future.result()
                        if temp_files:
                            processed_files.append(input_file)
                            self.processed_files.add(str(input_file))
                        else:
                            failed_files.append(input_file)
                    except Exception as e:
                        self.logger.error(f"Error processing {input_file}: {str(e)}")
                        failed_files.append(input_file)
                    pbar.update(1)

        self.logger.info(f"Successfully processed {len(processed_files)} files")
        if failed_files:
            self.logger.warning(f"Failed to process {len(failed_files)} files:")
            for failed_file in failed_files:
                self.logger.warning(f"  - {failed_file}")

        if processed_files:
            try:
                self.merge_temp_files()
                self.logger.info("Starting duplicate check after merging...")
                self.check_and_clean_duplicates()
            except Exception as e:
                self.logger.error(f"Error during merge phase: {str(e)}", exc_info=True)
                raise

        self.save_processed_files()

        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.logger.info("Temporary directory cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not remove temp directory: {str(e)}")

        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"Processing completed! Total duration: {duration}")

        return failed_files


if __name__ == "__main__":
    splitter = IncrementalDatasetSplitter(
        input_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed",
        output_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split",
        num_workers=24
    )
    splitter.process_all_files()