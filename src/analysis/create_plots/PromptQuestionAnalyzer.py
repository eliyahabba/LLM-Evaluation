import os
import time
from typing import List

import pandas as pd
import plotly.express as px


class PromptQuestionAnalyzer:
    def process_and_visualize_questions(
            self,
            df: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            interesting_datasets: List[str],
            base_results_dir: str,
    ) -> None:
        """
        Process question-level data (which template answered each sample_index correctly),
        compute the fraction of templates that answered each question correctly, and visualize
        the distribution (histogram) of those correctness fractions. Optionally,
        compute the same statistics across specified axes (e.g., template, separator, enumerator, choices_order).

        Args:
            df: DataFrame containing the raw data.
                Expected columns:
                    - 'dataset'
                    - 'template', 'separator', 'enumerator', 'choices_order' (or a subset of these)
                    - 'sample_index'
                    - 'score' (0 or 1)
            model_name: Name of the model being analyzed.
            shots_selected: Number of shots used in the experiment.
            interesting_datasets: List of dataset names to include in analysis.
            base_results_dir: Base directory for saving results (tables, plots).
            axes: List of axes (column names) for further breaking down question statistics.
                  For example: ["template", "separator", "enumerator", "choices_order"].
                  If None or empty, only overall question-level analysis will be performed.
        """
        start_time = time.time()
        print(f"Starting question-level analysis for model: {model_name}, shots={shots_selected}")

        # 1) Filter by the specified interesting datasets
        df_filtered = df[df["dataset"].isin(interesting_datasets)].copy()
        if df_filtered.empty:
            print("No data after filtering with interesting_datasets. Exiting.")
            return

        # Prepare a directory to save results
        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_'),
        )
        os.makedirs(model_dir, exist_ok=True)
        for dataset in interesting_datasets:
            dataset_df = df[df["dataset"] == dataset].copy()
            if dataset_df.empty:
                print(f"No data for dataset {dataset}. Skipping.")
                continue
        # 2) Compute accuracy for each question (without grouping by additional axes)
            overall_question_stats = self._aggregate_samples_accuracy(dataset_df)

            # 3) Save general tables and plots (no axis grouping)
            # Save tables for each dataset
            self._save_samples_accuracy_tables(
                overall_question_stats,
                model_dir,
                suffix="overall"
            )

            # 4) Compute accuracy grouped by specified axes, if provided
            dimensions = ["template", "separator", "enumerator", "choices_order"]
            for dim in dimensions:
                print(f"Computing question accuracy grouped by '{dim}'...")
                question_stats_by_axis = self._aggregate_samples_accuracy_by_axis(
                    df_filtered,
                    dim
                )
                # Save tables
                self._save_samples_accuracy_tables(
                    question_stats_by_axis,
                    model_dir,
                    dim,
                    suffix=f"by_{dim}"
                )

        total_time = time.time() - start_time
        print(f"Total question-level processing time for {model_name}: {total_time:.2f} seconds")
        print("-" * 50)

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

    def _aggregate_samples_accuracy_by_axis(self, df: pd.DataFrame, axis_name: str) -> pd.DataFrame:
        """
        Computes question-level accuracy grouped by an additional column (axis_name).
        For example, grouping by (dataset, sample_index, template) to see
        fraction_correct per question and per template.

        Returns a DataFrame with columns:
            - dataset
            - sample_index
            - axis_name (e.g., template)
            - sum_correct
            - total_configurations
            - fraction_correct
        """
        agg_start = time.time()

        # groupby includes the axis_name column
        grouped = df.groupby(["dataset", "sample_index", axis_name])["score"]
        sum_correct = grouped.sum()
        total_configurations = grouped.count()

        question_stats = pd.DataFrame({
            "sum_correct": sum_correct,
            "total_configurations": total_configurations
        }).reset_index()
        question_stats["fraction_correct"] = question_stats["sum_correct"] / question_stats["total_configurations"]

        print(f"Question-level grouping by axis='{axis_name}' completed in {time.time() - agg_start:.2f} seconds")
        return question_stats

    # ------------------------------------------------------------------------------------
    # Helper Functions for Saving
    # ------------------------------------------------------------------------------------
    def _save_samples_accuracy_tables(
            self,
            data: pd.DataFrame,
            model_dir: str,
            dim_subdir: str = None,
            suffix: str="overall"
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

        # Save a file for each dataset
        for dataset in data["dataset"].unique():
            subset = data[data["dataset"] == dataset].copy()
            if subset.empty:
                continue

            dataset_dir = os.path.join(model_dir, dataset.replace('/', '_'))
            os.makedirs(dataset_dir, exist_ok=True)
            if dim_subdir:
                dataset_dir = os.path.join(dataset_dir, dim_subdir)
                os.makedirs(dataset_dir, exist_ok=True)

            parquet_file = os.path.join(dataset_dir, f"samples_accuracy.parquet")
            subset.to_parquet(parquet_file, index=False)
            print(f"Saved question accuracy table (suffix={suffix}) for dataset {dataset} to {parquet_file}")

            self._plot_histograms_per_dataset(dataset, subset, dataset_dir, suffix)

    # ------------------------------------------------------------------------------------
    # Helper Functions for Visualization
    # ------------------------------------------------------------------------------------
    def _plot_histograms_per_dataset(
            self,
            dataset: str,
            subset: pd.DataFrame,
            plot_dir: str,
            suffix: str
    ) -> None:
        """
        For each dataset in 'data', create a histogram of fraction_correct and save it as HTML
        (using Plotly Express) in a subfolder named with the suffix.

        Args:
            data: DataFrame with columns [dataset, sample_index, fraction_correct, ...].
            model_dir: Root directory for the model results.
            suffix: A string to differentiate the type of grouping ("overall" or "by_{axis}").
        """

        fig_title = f"Distribution of Fraction Correct (suffix={suffix}) - {dataset}"
        fig = px.histogram(
            subset,
            x="fraction_correct",
            nbins=20,  # Number of bins in the histogram; adjustable
            title=fig_title,
            labels={"fraction_correct": "Fraction Correct"}
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1]),
            yaxis_title="Count of Questions",
        )

        fig_file = os.path.join(plot_dir, f"fraction_correct_histogram_{suffix}.html")
        fig.write_html(fig_file)
