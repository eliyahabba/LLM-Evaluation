import os
from pathlib import Path
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# הגדרת Logger בסיסי
logger = logging.getLogger("dedup_logger")
logger.setLevel(logging.INFO)

# הוספת StreamHandler (כדי להציג לוגים במסך) אם אין עדיין
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def process_parquet_file(parquet_file: Path, input_path: Path, dedup_eval_id_dir: Path, all_cols_dedup_dir: Path):
    """
    פונקציה המטפלת בקובץ פרקט בודד:
    1. קוראת את הקובץ.
    2. יוצרת שתי גרסאות:
       א. Dedup לפי evaluation_id
       ב. Dedup לפי כל העמודות (חוץ מ-"index")
    3. שומרת את שתי הגרסאות בתיקיות המקבילות
    4. מחזירה מידע לוגי (כמות כפילויות וכו') כדי שנוכל להדפיס אם רוצים.
    """
    try:
        logger.info(f"[START] - מתחיל לעבד את הקובץ: {parquet_file}")

        # קריאת הקובץ
        df = pl.read_parquet(str(parquet_file))
        logger.info(f"  נקראו {df.shape[0]} שורות ו-{df.shape[1]} עמודות מהקובץ {parquet_file}")

        # 1) הסרת כפילויות לפי evaluation_id
        df_eval_id_dedup = df.unique(subset=["evaluation_id"])
        logger.info(f"  Dedup לפי 'evaluation_id': שורות נותרו {df_eval_id_dedup.shape[0]} מתוך {df.shape[0]}")

        # 2) הסרת כפילויות לפי כל העמודות (פרט לעמודת 'index' אם קיימת)
        exclude_cols = ["index"] if "index" in df.columns else []
        all_cols_except_index = [c for c in df.columns if c not in exclude_cols]

        total_rows = df.shape[0]
        distinct_by_all = df.unique(subset=all_cols_except_index).shape[0]
        duplicates_count_all_cols = total_rows - distinct_by_all
        df_all_cols_dedup = df.unique(subset=all_cols_except_index)

        logger.info(f"  Dedup לפי כל העמודות (למעט {exclude_cols}): "
                    f"נמצאו {duplicates_count_all_cols} כפילויות, "
                    f"נותרו {df_all_cols_dedup.shape[0]} שורות")

        # 3) שמירה במבנה תיקיות מקביל
        relative_path = parquet_file.relative_to(input_path)

        # א) קובץ dedup לפי evaluation_id
        eval_id_file_path = dedup_eval_id_dir / relative_path
        eval_id_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_eval_id_dedup.write_parquet(str(eval_id_file_path))
        logger.info(f"  נשמר קובץ dedup לפי 'evaluation_id' ב: {eval_id_file_path}")

        # ב) קובץ dedup לפי כל העמודות
        all_cols_file_path = all_cols_dedup_dir / relative_path
        all_cols_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_all_cols_dedup.write_parquet(str(all_cols_file_path))
        logger.info(f"  נשמר קובץ dedup לפי כל העמודות ב: {all_cols_file_path}")

        logger.info(f"[DONE] - סיום עיבוד הקובץ: {parquet_file}\n")
        return parquet_file, total_rows, duplicates_count_all_cols

    except Exception as e:
        logger.error(f"[ERROR] בעת עיבוד הקובץ {parquet_file}: {e}", exc_info=True)
        return parquet_file, None, None, str(e)


def process_parquet_files_parallel(input_dir: str, max_workers: int = 4):
    """
    1. יצירת שתי תיקיות פלט (אחיות לתיקייה הראשית):
       - <input_dir>_eval_id_deduped
       - <input_dir>_all_cols_deduped
    2. חיפוש רקורסיבי של קבצי Parquet.
    3. עיבודם במקביל ע"י ThreadPoolExecutor (במספר workerים שהגדרנו).
    """
    input_path = Path(input_dir).resolve()

    # נגדיר נתיבי פלט
    dedup_eval_id_dir = input_path.parent / (input_path.name + "_eval_id_deduped")
    all_cols_dedup_dir = input_path.parent / (input_path.name + "_all_cols_deduped")
    dedup_eval_id_dir.mkdir(parents=True, exist_ok=True)
    all_cols_dedup_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_path.rglob("*.parquet"))
    if not parquet_files:
        logger.warning(f"[WARNING] No .parquet files found under {input_dir}")
        return

    logger.info(f"נמצאו {len(parquet_files)} קבצי Parquet בתיקייה {input_dir}")
    logger.info(f"מתחיל עיבוד מקבילי (max_workers={max_workers})\n")

    # עיבוד מקבילי של הקבצים
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_parquet_file, pf, input_path, dedup_eval_id_dir, all_cols_dedup_dir)
            for pf in parquet_files
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # הדפסת סיכום עיבוד
    logger.info("=== סיכום עיבוד ===")
    for item in results:
        if len(item) == 4 and item[3] is not None:
            # יש שגיאה
            logger.error(f"[FAILED]  קובץ: {item[0]}, שגיאה: {item[3]}")
        else:
            parquet_file, total_rows, duplicates_count_all_cols = item
            logger.info(
                f"[OK] - {parquet_file}\n"
                f"       סה\"כ שורות: {total_rows}, כפילויות לפי כל העמודות: {duplicates_count_all_cols}"
            )

    logger.info("\nDone processing all Parquet files in parallel!\n")


if __name__ == "__main__":
    input_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split"
    process_parquet_files_parallel(input_dir, max_workers=24)