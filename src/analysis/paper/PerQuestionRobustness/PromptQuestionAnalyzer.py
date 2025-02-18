import os
import time

import pandas as pd


class PromptQuestionAnalyzer:
    def process_and_visualize_questions_all_models(
            self,
            df: pd.DataFrame,
            dataset: str,
            base_results_dir: str,
    ) -> None:
        # Prepare a directory to save results
        model_dir = os.path.join(
            base_results_dir,
            f"Shots",
            "Models",
        )
        os.makedirs(model_dir, exist_ok=True)
        # נשתמש ב-value_counts כדי לספור את המופעים
        sample_counts = df.groupby('model')['sample_index'].value_counts()

        # נארגן את זה בצורת טבלה
        pivot_counts = sample_counts.unstack('model', fill_value=0)

        # נמצא את ה-sample_index שמופיעים מספיק פעמים בכל המודלים
        valid_samples = pivot_counts[pivot_counts.min(axis=1) >= 3000].index

        # נסנן את הדאטה המקורי
        df_filtered = df[df['sample_index'].isin(valid_samples)]

        # take only questions with total_configurations > 3000 for each model in model column
        overall_question_stats = self._aggregate_samples_accuracy(df_filtered)
        os.makedirs(model_dir, exist_ok=True)
        parquet_file = os.path.join(model_dir, f"{dataset}_samples_accuracy_3_models.parquet")
        overall_question_stats.to_parquet(parquet_file, index=False)
        # print the min of total_configurations and the dataset name

    def process_and_visualize_questions(
            self,
            df: pd.DataFrame,
            model_name: str,
            dataset: str,
            base_results_dir: str,
    ) -> None:
        start_time = time.time()
        print(f"Starting question-level analysis for model: {model_name}")

        # Prepare a directory to save results
        model_dir = os.path.join(
            base_results_dir,
            f"Shots",
            model_name.replace('/', '_'),
        )
        os.makedirs(model_dir, exist_ok=True)

        self._process_accuracy_stats(df, model_dir, dataset)

        total_time = time.time() - start_time
        print(f"Total question-level processing time for {model_name}: {total_time:.2f} seconds")
        print("-" * 50)

    def _process_accuracy_stats(
            self,
            dataset_df: pd.DataFrame,
            model_dir: str,
            dataset: str
    ) -> None:
        """Process accuracy statistics for a dataset."""
        overall_question_stats = self._aggregate_samples_accuracy(dataset_df)
        self._save_samples_accuracy_tables(overall_question_stats, model_dir, dataset)

    # ------------------------------------------------------------------------------------
    # Helper Functions for Aggregation
    # ------------------------------------------------------------------------------------
    def _aggregate_samples_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes, for each dataset+sample_index, how many templates answered correctly,
        how many total attempts, and the fraction correct.

        Returns a DataFrame with columns:
            - dataset
            - sample_index
            - sum_correct
            - total_configurations
            - fraction_correct
        """
        agg_start = time.time()

        grouped = df.groupby(["dataset", "sample_index"])["score"]
        sum_correct = grouped.sum()  # Number of templates that answered correctly
        total_configurations = grouped.count()  # Total number of templates that attempted

        question_stats = pd.DataFrame({
            "sum_correct": sum_correct,
            "total_configurations": total_configurations
        }).reset_index()
        question_stats["fraction_correct"] = (
                question_stats["sum_correct"] / question_stats["total_configurations"]).round(2)

        print(f"Overall question-level grouping completed in {time.time() - agg_start:.2f} seconds")
        return question_stats

    def _save_samples_accuracy_tables(
            self,
            data: pd.DataFrame,
            model_dir: str,
            dataset: str
    ) -> None:
        """
        Saves the question-accuracy DataFrame(s) in parquet format,
        one file per dataset, under a sub-folder named according to `suffix`.

        Args:
            data: DataFrame containing columns such as:
                  ['dataset', 'sample_index', 'sum_correct', 'total_configurations', 'fraction_correct']
                  (and possibly an extra column if grouping by an axis, e.g., template).
            model_dir: Main directory for the model results. Subfolders will be created for these tables.
            suffix: A string added to the subfolder/file name to indicate
                    whether it's "overall" or "by_{axis}".
        """
        os.makedirs(model_dir, exist_ok=True)
        parquet_file = os.path.join(model_dir, f"{dataset}_samples_accuracy.parquet")
        data.to_parquet(parquet_file, index=False)
        # print the min of total_configurations and the dataset name
        print(f"Min total_configurations for {dataset}: {data['total_configurations'].min()}")
