import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


class DatasetSplitter:
    def __init__(self, input_dir: str, output_dir: str):
        """
        :param input_dir: הנתיב לתיקייה שמכילה את קבצי הparquet המקוריים
        :param output_dir: הנתיב לתיקייה שבה יישמרו הקבצים המפוצלים
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_filename(self, model: str, shots: int, dataset: str) -> str:
        """יוצר שם קובץ עבור השלישייה של מודל-שוטס-דאטאסט"""
        return f"{model}_shots{shots}_{dataset}.parquet"

    def process_single_file(self, input_file: Path):
        """מעבד קובץ parquet בודד ומפצל אותו לקבצים לפי מודל/שוטס/דאטאסט"""
        print(f"Processing file: {input_file}")

        # קורא את הקובץ בצורה חסכונית בזיכרון
        parquet_file = pq.ParquetFile(input_file)

        # קורא את הדאטה בחלקים
        chunk_size = 10000  # אפשר לשנות את הגודל לפי הצורך
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df = batch.to_pandas()

            # מקבץ את הדאטה לפי השלישייה
            grouped = df.groupby(['model', 'shots', 'dataset'])

            for (model, shots, dataset), group_df in grouped:
                output_file = self.output_dir / self.get_output_filename(model, shots, dataset)

                # אם הקובץ כבר קיים, מוסיף את הדאטה החדשה
                if output_file.exists():
                    existing_df = pd.read_parquet(output_file)
                    combined_df = pd.concat([existing_df, group_df], ignore_index=True)
                    combined_df.to_parquet(output_file, index=False)
                else:
                    group_df.to_parquet(output_file, index=False)

    def process_all_files(self):
        """מעבד את כל קבצי ה-parquet בתיקיית הקלט"""
        parquet_files = list(self.input_dir.glob("*.parquet"))
        total_files = len(parquet_files)

        for i, file in enumerate(parquet_files, 1):
            print(f"Processing file {i}/{total_files}: {file.name}")
            try:
                self.process_single_file(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")

            # שחרור זיכרון
            import gc
            gc.collect()


# שימוש בקוד
if __name__ == "__main__":
    splitter = DatasetSplitter(
        input_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed",
        output_dir="/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split"
    )
    splitter.process_all_files()