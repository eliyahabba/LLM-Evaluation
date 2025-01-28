import concurrent
import logging
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


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
        """מעבד קובץ בודד ומחזיר רשימה של קבצים זמניים או רשימה ריקה אם נכשל"""
        worker_id = uuid.uuid4()
        temp_files = []

        self.logger.info(f"Starting to process file: {input_file}")
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            parquet_file = pq.ParquetFile(input_file)
            total_rows = parquet_file.metadata.num_rows

            with tqdm(total=total_rows, desc=f"Processing {input_file.name}", unit="rows") as pbar:
                for batch in parquet_file.iter_batches(batch_size=10000):
                    try:
                        df = batch.to_pandas()
                        grouped = df.groupby(['model', 'shots', 'dataset'])

                        for (model, shots, dataset), group_df in grouped:
                            try:
                                temp_file = self.temp_dir / f"{worker_id}_{model}_shots{shots}_{dataset}.parquet"
                                os.makedirs(temp_file.parent, exist_ok=True)
                                group_df.to_parquet(temp_file, index=False)
                                temp_files.append(str(temp_file))
                            except Exception as e:
                                self.logger.error(
                                    f"Error saving group {model}_{shots}_{dataset} from {input_file}: {str(e)}")
                                # ממשיך לקבוצה הבאה
                                continue

                        pbar.update(len(df))
                    except Exception as e:
                        self.logger.error(f"Error processing batch from {input_file}: {str(e)}")
                        # ממשיך לבאצ' הבא
                        continue

            self.logger.info(f"Completed processing file: {input_file}")
            return temp_files

        except Exception as e:
            self.logger.error(f"Fatal error processing file {input_file}: {str(e)}", exc_info=True)
            # מנקה קבצים זמניים שנוצרו אם יש כאלה
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up temp file {temp_file}: {str(cleanup_error)}")
            return []

    def process_all_files(self):
        """מעבד את כל הקבצים עם טיפול שגיאות משופר"""
        self.logger.info("Starting processing all files")
        start_time = datetime.now()

        parquet_files = list(self.input_dir.glob("*.parquet"))[:1]
        self.logger.info(f"Found {len(parquet_files)} parquet files to process")

        failed_files = []
        processed_files = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {executor.submit(self.process_single_file, f): f for f in parquet_files}

            with tqdm(total=len(parquet_files), desc="Processing files", unit="file") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    input_file = future_to_file[future]
                    try:
                        temp_files = future.result()
                        if temp_files:  # אם הקובץ עובד בהצלחה
                            processed_files.append(input_file)
                        else:  # אם הקובץ נכשל
                            failed_files.append(input_file)
                    except Exception as e:
                        self.logger.error(f"Error processing {input_file}: {str(e)}")
                        failed_files.append(input_file)
                    pbar.update(1)

        # סיכום התוצאות
        self.logger.info(f"Successfully processed {len(processed_files)} files")
        if failed_files:
            self.logger.warning(f"Failed to process {len(failed_files)} files:")
            for failed_file in failed_files:
                self.logger.warning(f"  - {failed_file}")

        # ממשיך למיזוג רק אם יש קבצים שעובדו בהצלחה
        if processed_files:
            try:
                self.merge_temp_files()
            except Exception as e:
                self.logger.error(f"Error during merge phase: {str(e)}", exc_info=True)
                raise

        # ניקוי
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.logger.info("Temporary directory cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not remove temp directory: {str(e)}")

        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"Processing completed! Total duration: {duration}")

        # מחזיר רשימת הקבצים שנכשלו כדי שאפשר יהיה לנסות אותם שוב
        return failed_files

    def merge_temp_files(self):
        """ממזג את הקבצים הזמניים לפי שלישיות של model-shots-dataset"""
        self.logger.info("Starting to merge temporary files")

        # מיפוי של כל הקבצים הזמניים לפי השלישייה שלהם
        triplet_files_map = {}

        # בודק כל קובץ בתיקייה הזמנית
        for temp_file in self.temp_dir.glob("*"):
            try:
                # בודק אם יש קובץ parquet בתוך התיקייה
                if temp_file.is_dir():
                    parquet_files = list(temp_file.glob("*.parquet"))
                    if not parquet_files:
                        self.logger.warning(f"No parquet files found in directory {temp_file}")
                        continue

                    # קורא את הדאטה כדי לחלץ את השלישייה
                    df = pd.read_parquet(parquet_files[0])
                    if len(df) == 0:
                        self.logger.warning(f"Empty dataframe in {temp_file}")
                        continue

                    # מחלץ את הערכים של השלישייה מהשורה הראשונה
                    model = df.iloc[0]['model']
                    shots = df.iloc[0]['shots']
                    dataset = df.iloc[0]['dataset']

                    triplet_key = f"{model}_shots{shots}_{dataset}"

                    if triplet_key not in triplet_files_map:
                        triplet_files_map[triplet_key] = []
                    triplet_files_map[triplet_key].append(temp_file)

            except Exception as e:
                self.logger.error(f"Error processing temp file/dir {temp_file}: {str(e)}")

        self.logger.info(f"Found {len(triplet_files_map)} unique triplets to merge")

        if not triplet_files_map:
            self.logger.error("No triplets found to merge! Printing all files in temp directory:")
            for f in self.temp_dir.glob("*"):
                self.logger.error(f"File in temp dir: {f}")
            return

        # מיזוג הקבצים לפי השלישיות
        with tqdm(total=len(triplet_files_map), desc="Merging files by triplet") as pbar:
            for triplet_key, temp_dirs in triplet_files_map.items():
                try:
                    self.logger.info(f"Processing triplet {triplet_key} with {len(temp_dirs)} files")

                    # קורא את כל הקבצים של השלישייה הזו
                    dfs = []
                    for temp_dir in temp_dirs:
                        try:
                            parquet_files = list(temp_dir.glob("*.parquet"))
                            if parquet_files:
                                df = pd.read_parquet(parquet_files[0])
                                dfs.append(df)
                                self.logger.info(f"Successfully read {temp_dir} with {len(df)} rows")
                        except Exception as e:
                            self.logger.error(f"Error reading from directory {temp_dir}: {str(e)}")

                    if not dfs:
                        self.logger.error(f"No valid data found for triplet {triplet_key}")
                        continue

                    # ממזג את כל הדאטה של השלישייה
                    combined_df = pd.concat(dfs, ignore_index=True)
                    output_file = self.output_dir / f"{triplet_key}.parquet"
                    os.makedirs(output_file.parent, exist_ok=True)
                    combined_df.to_parquet(output_file, index=False)
                    self.logger.info(f"Successfully created {output_file} with {len(combined_df)} rows")

                    # מוחק את התיקיות הזמניות
                    for temp_dir in temp_dirs:
                        try:
                            if temp_dir.is_dir():
                                shutil.rmtree(temp_dir)
                            else:
                                temp_dir.unlink()
                            self.logger.info(f"Deleted temporary directory {temp_dir}")
                        except Exception as e:
                            self.logger.error(f"Error deleting temp directory {temp_dir}: {str(e)}")

                except Exception as e:
                    self.logger.error(f"Error processing triplet {triplet_key}: {str(e)}", exc_info=True)

                pbar.update(1)
    def process_all_files2(self):
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
        num_workers=24
    )
    splitter.process_all_files()
