import concurrent
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Set, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


class HFDatasetSplitter:
    def __init__(self, repo_id: str, output_dir: str, num_workers: int = 4):
        self.repo_id = repo_id
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.processed_files_path = self.output_dir / "processed_files.json"
        self.hf_api = HfApi()

        # Create a temporary directory that will be automatically cleaned up
        self.temp_dir = Path("/cs/snapless/gabis/eliyahabba/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique subdirectory for this instance
        self.temp_dir = self.temp_dir / str(uuid.uuid4())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.setup_logger()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load previously processed files
        self.processed_files = self.load_processed_files()

        self.logger.info(f"Initialized splitter with {num_workers} workers")
        self.logger.info(f"Repository ID: {repo_id}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Temporary directory: {self.temp_dir}")
        self.logger.info(f"Found {len(self.processed_files)} previously processed files")

    def load_processed_files(self) -> Set[str]:
        """Load the set of previously processed files."""
        if self.processed_files_path.exists():
            try:
                with open(self.processed_files_path, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                self.logger.error(f"Error loading processed files list: {e}")
                return set()
        return set()

    def save_processed_files(self):
        """Save the current set of processed files."""
        try:
            with open(self.processed_files_path, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            self.logger.error(f"Error saving processed files list: {e}")

    def get_new_files(self) -> List[str]:
        """Get list of new files from HF that haven't been processed yet."""
        try:
            all_files = set(self.hf_api.list_repo_files(self.repo_id, repo_type="dataset"))
            # Filter for parquet files only
            parquet_files = {f for f in all_files if f.endswith('.parquet')}
            new_files = parquet_files - self.processed_files
            return list(new_files)
        except Exception as e:
            self.logger.error(f"Error fetching files from Hugging Face: {e}")
            return []

    def download_file(self, file_path: str) -> Optional[Path]:
        """Download a single file from HF to the temporary directory."""
        try:
            local_path = self.temp_dir / Path(file_path).name
            self.logger.info(f"Downloading file {file_path} to {local_path}")
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=file_path,
                local_dir=self.temp_dir,
                etag_timeout=3600
            )
            return local_path
        except Exception as e:
            self.logger.error(f"Error downloading file {file_path}: {e}")
            return None

    def setup_logger(self):
        self.logger = logging.getLogger('HFDatasetSplitter')
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

    def effie_process_single_file(self, file_path: str) -> List[str]:
        worker_id = uuid.uuid4()
        self.logger.info(f"Starting to process file: {file_path}")

        # Download the file
        local_file = self.download_file(file_path)
        if not local_file:
            self.logger.error(f"Failed to download {file_path}")
            return []


        temp_file_handles = {}  # key: group key, value: True if file exists
        temp_files = set()

        if not local_file:
            self.logger.error(f"Failed to download {file_path}")
            return []
        try:

            parquet_file = pq.ParquetFile(local_file)
            total_rows = parquet_file.metadata.num_rows
            with tqdm(total=total_rows, desc=f"Processing {local_file.name}", unit="rows") as pbar:
                for batch in parquet_file.iter_batches(batch_size=100000):
                    # Convert the record batch to a pyarrow Table
                    table = pa.Table.from_batches([batch])

                    # Extract nested fields using arrow's field extraction.
                    # These calls assume that 'model', 'prompt_config', and 'instance' are struct columns.
                    # model_name = table.column("model").field("model_info").field("name")
                    model_name = table.column("model").combine_chunks().field("model_info").field("name")
                    # shots = table.column("prompt_config").field("format").field("shots")
                    shots = table.column("prompt_config").combine_chunks().field("format").field("shots")
                    # dataset_name = table.column("instance").field("sample_identifier").field("dataset_name")
                    dataset_name = table.column("instance").combine_chunks().field("sample_identifier").field("dataset_name")

                    # Append the new columns to the table.
                    table = table.append_column("model_name", model_name)
                    table = table.append_column("shots", shots)
                    table = table.append_column("dataset_name", dataset_name)

                    # Create a struct of the three new columns to get unique group combinations.
                    group_struct = pa.StructArray.from_arrays(
                        [model_name, shots, dataset_name],
                        names=["model_name", "shots", "dataset_name"]
                    )
                    unique_groups = pc.unique(group_struct)

                    # Process each unique group within the batch.
                    for group in unique_groups.to_pylist():
                        m = group["model_name"]
                        s = group["shots"]
                        d = group["dataset_name"]
                        key = f"{worker_id}_{m}_shots{s}_{d}"
                        temp_file = self.temp_dir / f"{key}.parquet"
                        temp_files.add(str(temp_file))

                        # Build the filter mask: (model_name == m) & (shots == s) & (dataset_name == d)
                        mask = pc.and_(
                            pc.and_(pc.equal(model_name, m), pc.equal(shots, s)),
                            pc.equal(dataset_name, d)
                        )
                        filtered_table = table.filter(mask)

                        # Write the filtered table to the appropriate file.
                        if key not in temp_file_handles:
                            os.makedirs(temp_file.parent, exist_ok=True)
                            pq.write_table(filtered_table, temp_file)
                            temp_file_handles[key] = True
                        else:
                            pq.write_table(filtered_table, temp_file, append=True)

                    pbar.update(table.num_rows)
        except Exception as e:
            self.logger.error(f"Fatal error processing file {file_path}: {str(e)}", exc_info=True)
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up temp file {temp_file}: {str(cleanup_error)}")
            return []

        return list(temp_files)

    def process_single_file(self, file_path: str) -> List[str]:
        """Process a single file after downloading it."""
        worker_id = uuid.uuid4()
        temp_files = set()
        temp_file_handles = {}  # Dictionary to store open file handles

        self.logger.info(f"Starting to process file: {file_path}")

        # Download the file
        local_file = self.download_file(file_path)
        if not local_file:
            self.logger.error(f"Failed to download {file_path}")
            return []

        try:
            parquet_file = pq.ParquetFile(local_file)
            total_rows = parquet_file.metadata.num_rows

            with tqdm(total=total_rows, desc=f"Processing {local_file.name}", unit="rows") as pbar:
                for batch in parquet_file.iter_batches(batch_size=10000):  # Reduced batch size
                    df = batch.to_pandas()
                    df['model_name'] = [x['model_info']['name'] for x in df['model']]
                    df['shots'] = [x['format']['shots'] for x in df['prompt_config']]
                    df['dataset_name'] = [x['sample_identifier']['dataset_name'] for x in df['instance']]

                    # Group by extracted fields
                    grouped = df.groupby(['model_name', 'shots', 'dataset_name'])

                    for (model, shots, dataset), group_df in grouped:
                        key = f"{worker_id}_{model}_shots{shots}_{dataset}"
                        temp_file = self.temp_dir / f"{key}.parquet"
                        temp_files.add(str(temp_file))

                        if key not in temp_file_handles:
                            # Create directory if it doesn't exist
                            os.makedirs(temp_file.parent, exist_ok=True)
                            # Write the first chunk with the schema
                            group_df.to_parquet(temp_file, index=False)
                        else:
                            # Append to existing file without writing schema
                            group_df.to_parquet(
                                temp_file,
                                index=False,
                                append=True
                            )

                    pbar.update(len(df))
                    # Clear memory explicitly
                    del df
                    del grouped

            # Clean up the downloaded file
            local_file.unlink()

        except Exception as e:
            self.logger.error(f"Fatal error processing file {file_path}: {str(e)}", exc_info=True)
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up temp file {temp_file}: {str(cleanup_error)}")
            return []

        return list(temp_files)

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
            future_to_file = {executor.submit(self.effie_process_single_file, f): f for f in new_files}

            with tqdm(total=len(new_files), desc="Processing files", unit="file") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    input_file = future_to_file[future]
                    try:
                        temp_files = future.result()
                        if temp_files:
                            processed_files.append(input_file)
                            # Add to processed files set
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
            except Exception as e:
                self.logger.error(f"Error during merge phase: {str(e)}", exc_info=True)
                raise

        # Save the updated processed files list
        self.save_processed_files()

        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
            self.logger.info("Temporary directory cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not remove temp directory: {str(e)}")

        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"Processing completed! Total duration: {duration}")

        return failed_files

    def merge_temp_files(self):
        """Merge temporary files, now considers existing output files."""
        self.logger.info("Starting to merge temporary files")

        triplet_files_map = {}

        for temp_file in self.temp_dir.glob("*"):
            try:
                if temp_file.is_dir():
                    for parquet_file in temp_file.glob("*.parquet"):
                        triplet_key = parquet_file.stem
                        if triplet_key not in triplet_files_map:
                            triplet_files_map[triplet_key] = []
                        triplet_files_map[triplet_key].append(parquet_file)
            except Exception as e:
                self.logger.error(f"Error processing temp file/dir {temp_file}: {str(e)}")

        self.logger.info(f"Found {len(triplet_files_map)} unique triplets to merge")

        with tqdm(total=len(triplet_files_map), desc="Merging files by triplet") as pbar:
            for triplet_key, temp_files in triplet_files_map.items():
                try:
                    self.logger.info(f"Processing triplet {triplet_key} with {len(temp_files)} files")

                    dfs = []

                    # Check if output file already exists
                    output_file = self.output_dir / f"{triplet_key}.parquet"
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
                        os.makedirs(output_file.parent, exist_ok=True)
                        combined_df.to_parquet(output_file, index=False)
                        self.logger.info(f"Successfully created/updated {output_file} with {len(combined_df)} rows")

                except Exception as e:
                    self.logger.error(f"Error processing triplet {triplet_key}: {str(e)}", exc_info=True)

                pbar.update(1)


if __name__ == "__main__":
    import os
    from datetime import datetime
    from concurrent.futures import ProcessPoolExecutor

    output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_split"
    os.makedirs(output_dir, exist_ok=True)
    splitter = HFDatasetSplitter(
        repo_id="eliyahabba/llm-evaluation",  # Replace with your HF repo ID
        output_dir=output_dir,
        num_workers=24
    )
    splitter.process_all_files()
