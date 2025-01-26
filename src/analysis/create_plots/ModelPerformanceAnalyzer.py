import os

import pandas as pd
import plotly.express as px


class ModelPerformanceAnalyzer:
    def generate_model_performance_comparison(
            self,
            df: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            base_results_dir: str
    ) -> None:
        """
        Generate performance comparison plots for multiple models across datasets.

        Args:
            df: DataFrame containing performance data with columns ['model', 'dataset', 'score'].
        """
        print("Starting performance comparison...")
        self.generate_model_performance_comparison_by_dataset(df, model_name, shots_selected, base_results_dir)
        self.generate_model_performance_comparison_all_datasets(df, model_name, shots_selected, base_results_dir)

    def generate_model_performance_comparison_all_datasets(
            self,
            df: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            base_results_dir: str
    ) -> None:
        grouped = df.groupby(["template", "separator", "enumerator", "choices_order"])
        sum_scores = grouped['score'].sum()
        n_samples = grouped['score'].count()

        # now for each row in the processed_df, we need to calculate the size of the configuration per dataset,
        processed_df = pd.DataFrame({
            'sum_scores': sum_scores,
            'count': n_samples,
        }).reset_index()

        processed_df["accuracy"] = processed_df["sum_scores"] / processed_df["count"]

        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )
        processed_df.to_parquet(os.path.join(model_dir, f"model_statistics_all_datasets.parquet"))

        return
        dataset_statistics = processed_df["accuracy"].agg(
            average_accuracy="mean",
            std_dev_accuracy="std",
            configuration_count="count"
        )

        dataset_statistics = dataset_statistics.round(4)

        dataset_statistics.to_parquet(os.path.join(model_dir, f"model_statistics_all_datasets.parquet"))
        # self.plot_model_performance_comparison(dataset_statistics, model_name, shots_selected, model_dir)

    def generate_model_performance_comparison_by_dataset(
            self,
            df: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            base_results_dir: str
    ) -> None:
        grouped = df.groupby(["dataset", "template", "separator", "enumerator", "choices_order"])
        sum_scores = grouped['score'].sum()
        n_samples = grouped['score'].count()

        processed_df = pd.DataFrame({
            'sum_scores': sum_scores,
            'count': n_samples
        }).reset_index()

        processed_df["accuracy"] = processed_df["sum_scores"] / processed_df["count"]

        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )
        os.makedirs(model_dir, exist_ok=True)
        processed_df.to_parquet(os.path.join(model_dir, f"model_statistics.parquet"))
        return

    def plot_model_performance_comparison(
            self,
            dataset_statistics: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            model_dir: str
    ) -> None:
        """
        Generate performance comparison plots with configurations count.

        Args:
            df: DataFrame containing performance data with columns ['model', 'dataset', 'score'].
            model_name: Name of the model being analyzed.
            shots_selected: Number of shots used in the experiment.
            base_results_dir: Directory to save the plots.
        """
        # Generate bar plot with error bars and configuration counts
        fig = px.bar(
            dataset_statistics,
            x='dataset',
            y='average_accuracy',
            error_y='std_dev_accuracy',
            title=f'Model Performance: {model_name} (Shots: {shots_selected})',
            labels={
                'dataset': 'Dataset',
                'average_a§ccuracy': 'Average Accuracy',
                'std_dev_accuracy': 'Standard Deviation'
            },
            text='configuration_count'  # Show number of configurations on bars
        )

        fig.update_layout(
            xaxis_tickangle=45,
            yaxis=dict(range=[0, 1.05]),
            showlegend=False,
            height=600,
            width=1000
        )

        # Format the configuration count text
        fig.update_traces(
            texttemplate='n=%{text}',  # Add 'n=' prefix to configuration count
            textposition='outside',  # Position text above bars
            textfont=dict(size=12)  # Adjust text size
        )

        # Save plot
        fig.write_html(os.path.join(model_dir, "performance_bar_plot.html"))

        print(f"Performance comparison completed for {model_name}. Results saved in {model_dir}")
