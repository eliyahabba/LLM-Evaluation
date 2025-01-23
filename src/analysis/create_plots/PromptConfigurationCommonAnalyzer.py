# processor.py
import os
import time
from typing import List, Set

import pandas as pd
import plotly.express as px


class PromptConfigurationAnalyzer:
    def process_and_visualize_configurations(
            self,
            df: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            interesting_datasets: List[str],
            base_results_dir: str
    ) -> Set[str]:
        """
        Process configuration data and generate visualization plots for model performance.

        Args:
            df: DataFrame containing the raw configuration performance data
            model_name: Name of the model being analyzed
            shots_selected: Number of shots used in the experiment
            interesting_datasets: List of datasets to include in analysis
            base_results_dir: Base directory for saving results
        """
        start_time = time.time()

        # Process the data
        grouped_data = self._aggregate_configuration_scores(df)
        filtered_datasets = self._filter_and_select_datasets(grouped_data, interesting_datasets)
        final_data = self._prepare_visualization_data(grouped_data, filtered_datasets)

        # Generate and save visualizations
        self._generate_plots(
            final_data,
            model_name,
            shots_selected,
            filtered_datasets,
            base_results_dir
        )

        total_time = time.time() - start_time
        print(f"Total processing time for {model_name}: {total_time:.2f} seconds")
        print("-" * 50)
        return filtered_datasets

    def _aggregate_configuration_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate scores for each configuration combination."""
        group_start = time.time()

        grouped = df.groupby(["dataset", "template", "separator", "enumerator", "choices_order"])
        sum_scores = grouped['score'].sum()
        n_samples = grouped['score'].count()

        processed_df = pd.DataFrame({
            'sum_scores': sum_scores,
            'count': n_samples
        }).reset_index()

        processed_df["accuracy"] = processed_df["sum_scores"] / processed_df["count"]

        print(f"Initial grouping completed in {time.time() - group_start:.2f} seconds")
        return processed_df


    def _filter_and_select_datasets(
            self,
            data: pd.DataFrame,
            interesting_datasets: List[str]
    ) -> Set[str]:
        """Filter configurations and select datasets for analysis."""
        # Filter configurations with exactly 100 examples
        filtered_df = data[data["count"] == 100].copy()

        # Get datasets with most configurations
        dataset_counts = (filtered_df.groupby("dataset")["count"]
                          .count()
                          .reset_index(name="num_quads"))

        top5_datasets = dataset_counts.nlargest(5, "num_quads")["dataset"].tolist()
        return set(interesting_datasets).union(top5_datasets)

    def _prepare_visualization_data(
            self,
            data: pd.DataFrame,
            selected_datasets: Set[str]
    ) -> pd.DataFrame:
        """Prepare data for visualization by filtering and formatting."""
        filtered_df = data[data["dataset"].isin(selected_datasets)].copy()

        # Create configuration string
        filtered_df["quad"] = (
                filtered_df["template"] + " | " +
                filtered_df["separator"] + " | " +
                filtered_df["enumerator"] + " | " +
                filtered_df["choices_order"]
        )

        return filtered_df

    def _generate_plots(
            self,
            data: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            datasets: Set[str],
            base_results_dir: str
    ) -> None:
        """Generate and save visualization plots."""
        # Prepare directory
        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )
        os.makedirs(model_dir, exist_ok=True)

        print(f"Generating and saving plots for model: {model_name}")
        plot_start = time.time()

        # Generate plot for each dataset
        for dataset in datasets:
            subset = data[data["dataset"] == dataset]
            if subset.empty:
                continue

            # fig = self._create_accuracy_plot(subset, dataset)
            dataset_dir = os.path.join(model_dir, f"{dataset.replace('/', '_')}")
            os.makedirs(dataset_dir, exist_ok=True)
            output_fig_file = os.path.join(dataset_dir, "prompt_configuration_analyzer.html")
            # fig.write_html(output_fig_file)
            # save also the data itself to parquet
            output_parquet_file = os.path.join(dataset_dir, "prompt_configuration_analyzer.parquet")
            subset.to_parquet(output_parquet_file, index=False)

        print(f"Plotting completed in {time.time() - plot_start:.2f} seconds")

    def _create_accuracy_plot(
            self,
            data: pd.DataFrame,
            dataset: str
    ):
        """Create accuracy plot for a specific dataset."""
        fig = px.scatter(
            data,
            x="quad",
            y="accuracy",
            text="count",
            title=f"Accuracy by Quad - {dataset}",
            labels={"quad": "Configuration", "accuracy": "Accuracy"}
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(range=[0, 1.05])
        )
        return fig