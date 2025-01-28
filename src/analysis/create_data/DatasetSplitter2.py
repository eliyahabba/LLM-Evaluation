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

        parquet_files = list(self.input_dir.glob("*.parquet"))[:2]
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
        """ממזג את הקבצים הזמניים"""
        self.logger.info("Starting to merge temporary files")
        temp_files_map = {}

        # מיפוי הקבצים הזמניים
        for temp_file in self.temp_dir.glob("*.parquet"):
            try:
                # מחלץ את השם ללא ה-UUID בהתחלה
                uuid_and_rest = temp_file.name.split('_', 1)  # מפצל רק פעם אחת כדי להפריד את ה-UUID
                if len(uuid_and_rest) < 2:
                    self.logger.warning(f"Unexpected filename format: {temp_file.name}")
                    continue

                final_name = uuid_and_rest[1]  # לוקח את כל מה שאחרי ה-UUID
                if final_name not in temp_files_map:
                    temp_files_map[final_name] = []
                temp_files_map[final_name].append(temp_file)

            except Exception as e:
                self.logger.error(f"Error processing temp file name {temp_file}: {str(e)}")
                continue

        self.logger.info(f"Found {len(temp_files_map)} unique combinations to merge")

        if len(temp_files_map) == 0:
            self.logger.error("No files found to merge! Printing all files in temp directory:")
            for f in self.temp_dir.glob("*"):
                self.logger.error(f"File in temp dir: {f}")
            return

        # מיזוג עם progress bar
        with tqdm(total=len(temp_files_map), desc="Merging files") as pbar:
            for final_name, temp_files in temp_files_map.items():
                try:
                    final_file = self.output_dir / final_name
                    self.logger.info(f"Merging {len(temp_files)} files into {final_file}")

                    dfs = []
                    for f in temp_files:
                        try:
                            df = pd.read_parquet(f)
                            dfs.append(df)
                        except Exception as e:
                            self.logger.error(f"Error reading temp file {f}: {str(e)}")
                            continue

                    if not dfs:  # אם לא הצלחנו לקרוא אף קובץ
                        self.logger.error(f"No valid dataframes to merge for {final_name}")
                        continue

                    combined_df = pd.concat(dfs, ignore_index=True)
                    combined_df.to_parquet(final_file, index=False)

                    # מחיקת קבצים זמניים רק אחרי שהמיזוג הצליח
                    for f in temp_files:
                        try:
                            f.unlink()
                        except Exception as e:
                            self.logger.error(f"Error deleting temp file {f}: {str(e)}")

                    self.logger.info(f"Successfully created {final_file} with {len(combined_df)} rows")

                except Exception as e:
                    self.logger.error(f"Error merging files for {final_name}: {str(e)}", exc_info=True)

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
