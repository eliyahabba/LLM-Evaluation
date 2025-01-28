import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
import uuid
from typing import List, Tuple
from tqdm import tqdm
import logging
from datetime import datetime


class ParallelDatasetSplitter:
    def __init__(self, input_dir: str, output_dir: str, num_workers: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"
        self.num_workers = num_workers

        # הגדרת הלוגר
        self.setup_logger()

        # יצירת התיקיות הנדרשות
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized splitter with {num_workers} workers")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")

    def setup_logger(self):
        """מגדיר את הלוגר"""
        self.logger = logging.getLogger('DatasetSplitter')
        self.logger.setLevel(logging.INFO)

        # יוצר קובץ לוג עם timestamp
        log_filename = f"split_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)

        # יוצר handler לקונסול
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # מגדיר פורמט
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_output_filename(self, model: str, shots: int, dataset: str) -> str:
        return f"{model}_shots{shots}_{dataset}.parquet"

    def process_single_file(self, input_file: Path) -> List[str]:
        """מעבד קובץ בודד"""
        worker_id = uuid.uuid4()
        temp_files = []

        self.logger.info(f"Starting to process file: {input_file}")
        try:
            parquet_file = pq.ParquetFile(input_file)
            total_rows = parquet_file.metadata.num_rows

            # יוצר progress bar לכל קובץ
            with tqdm(total=total_rows, desc=f"Processing {input_file.name}", unit="rows") as pbar:
                for batch in parquet_file.iter_batches(batch_size=100000):
                    df = batch.to_pandas()
                    grouped = df.groupby(['model', 'shots', 'dataset'])

                    for (model, shots, dataset), group_df in grouped:
                        temp_file = self.temp_dir / f"{worker_id}_{model}_shots{shots}_{dataset}.parquet"
                        os.makedirs(temp_file.parent, exist_ok=True)
                        group_df.to_parquet(temp_file, index=False)
                        temp_files.append(str(temp_file))

                    pbar.update(len(df))

            self.logger.info(f"Completed processing file: {input_file}")
            self.logger.info(f"Created {len(temp_files)} temporary files")

        except Exception as e:
            self.logger.error(f"Error processing file {input_file}: {str(e)}", exc_info=True)
            raise

        return temp_files

    def merge_temp_files(self):
        """ממזג את הקבצים הזמניים"""
        self.logger.info("Starting to merge temporary files")
        temp_files_map = {}

        # מיפוי הקבצים הזמניים
        for temp_file in self.temp_dir.glob("*.parquet"):
            parts = temp_file.name.split('_')
            if len(parts) >= 4:
                final_name = '_'.join(parts[1:])
                if final_name not in temp_files_map:
                    temp_files_map[final_name] = []
                temp_files_map[final_name].append(temp_file)

        self.logger.info(f"Found {len(temp_files_map)} unique combinations to merge")

        # מיזוג עם progress bar
        with tqdm(total=len(temp_files_map), desc="Merging files") as pbar:
            for final_name, temp_files in temp_files_map.items():
                try:
                    final_file = self.output_dir / final_name
                    self.logger.info(f"Merging {len(temp_files)} files into {final_file}")

                    dfs = [pd.read_parquet(f) for f in temp_files]
                    combined_df = pd.concat(dfs, ignore_index=True)
                    combined_df.to_parquet(final_file, index=False)

                    # מחיקת קבצים זמניים
                    for f in temp_files:
                        f.unlink()

                    self.logger.info(f"Successfully created {final_file} with {len(combined_df)} rows")

                except Exception as e:
                    self.logger.error(f"Error merging files for {final_name}: {str(e)}", exc_info=True)

                pbar.update(1)

    def process_all_files(self):
        """מעבד את כל הקבצים"""
        self.logger.info("Starting processing all files")
        start_time = datetime.now()

        parquet_files = list(self.input_dir.glob("*.parquet"))
        self.logger.info(f"Found {len(parquet_files)} parquet files to process")

        # עיבוד מקבילי
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            try:
                # מריץ את העיבוד עם progress bar
                list(tqdm(
                    executor.map(self.process_single_file, parquet_files),
                    total=len(parquet_files),
                    desc="Processing files",
                    unit="file"
                ))

            except Exception as e:
                self.logger.error(f"Error during parallel processing: {str(e)}", exc_info=True)
                raise

        # מיזוג הקבצים
        self.merge_temp_files()

        # ניקוי
        try:
            self.temp_dir.rmdir()
            self.logger.info("Temporary directory cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not remove temp directory: {str(e)}")

        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"Processing completed! Total duration: {duration}")


# שימוש
if __name__ == "__main__":
    splitter = ParallelDatasetSplitter(
        input_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed",
        output_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split",
        num_workers=4
    )
    splitter.process_all_files()