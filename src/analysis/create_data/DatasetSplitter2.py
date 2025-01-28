import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
import uuid
from typing import List, Tuple


class ParallelDatasetSplitter:
    def __init__(self, input_dir: str, output_dir: str, num_workers: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"
        self.num_workers = num_workers

        # יצירת התיקיות הנדרשות
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_output_filename(self, model: str, shots: int, dataset: str) -> str:
        """יוצר שם קובץ עבור השלישייה של מודל-שוטס-דאטאסט"""
        return f"{model}_shots{shots}_{dataset}.parquet"

    def process_single_file(self, input_file: Path) -> List[str]:
        """מעבד קובץ בודד ומחזיר רשימה של הקבצים הזמניים שנוצרו"""
        temp_files = []
        worker_id = uuid.uuid4()

        parquet_file = pq.ParquetFile(input_file)

        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()
            grouped = df.groupby(['model', 'shots', 'dataset'])

            for (model, shots, dataset), group_df in grouped:
                temp_file = self.temp_dir / f"{worker_id}_{model}_shots{shots}_{dataset}.parquet"
                os.makedirs(temp_file.parent, exist_ok=True)
                group_df.to_parquet(temp_file, index=False)
                temp_files.append(str(temp_file))

        return temp_files

    def merge_temp_files(self):
        """ממזג את כל הקבצים הזמניים לקבצים הסופיים"""
        # מיפוי של כל הקבצים הזמניים לפי השלישייה שלהם
        temp_files_map = {}

        for temp_file in self.temp_dir.glob("*.parquet"):
            # מחלץ את פרטי השלישייה מהשם
            parts = temp_file.name.split('_')
            if len(parts) >= 4:  # וידוא שיש מספיק חלקים בשם
                final_name = '_'.join(parts[1:])  # מסיר את ה-UUID
                if final_name not in temp_files_map:
                    temp_files_map[final_name] = []
                temp_files_map[final_name].append(temp_file)

        # מיזוג הקבצים
        for final_name, temp_files in temp_files_map.items():
            final_file = self.output_dir / final_name
            dfs = [pd.read_parquet(f) for f in temp_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_parquet(final_file, index=False)

            # מחיקת הקבצים הזמניים
            for f in temp_files:
                f.unlink()

    def process_all_files(self):
        """מעבד את כל הקבצים במקביל"""
        parquet_files = list(self.input_dir.glob("*.parquet"))

        # עיבוד מקבילי של הקבצים
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # מריץ את העיבוד במקביל
            results = list(executor.map(self.process_single_file, parquet_files))

        # מיזוג כל הקבצים הזמניים
        print("Merging temporary files...")
        self.merge_temp_files()

        # ניקוי תיקיית הטמפ
        self.temp_dir.rmdir()

        print("Processing completed!")


# שימוש
if __name__ == "__main__":
    splitter = ParallelDatasetSplitter(
        input_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed",
        output_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split",
        num_workers=4  # מספר התהליכים המקביליים
    )
    splitter.process_all_files()