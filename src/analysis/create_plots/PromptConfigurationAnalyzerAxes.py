# processor.py
import os
import time
from typing import List, Set

import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt


class PromptConfigurationAnalyzerAxes:
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

        This method first performs the original 'quad' analysis where all four columns
        (template, separator, enumerator, choices_order) are used together. Then it also
        calls a new analysis that looks at each of the four columns individually to see
        how they perform on average when the other three columns are "unified."

        Args:
            df: DataFrame containing the raw configuration performance data
            model_name: Name of the model being analyzed
            shots_selected: Number of shots used in the experiment
            interesting_datasets: List of datasets to include in analysis
            base_results_dir: Base directory for saving results
        """
        start_time = time.time()

        # -------------------------
        # 1) Process the 'quad' data (original logic)
        # -------------------------
        grouped_data = self._aggregate_configuration_scores(df)
        # filtered_datasets = self._filter_and_select_datasets(grouped_data, interesting_datasets)
        filtered_datasets = interesting_datasets
        final_data = self._prepare_visualization_data(grouped_data, filtered_datasets)
        # # Generate and save visualizations for the 'quad' analysis
        self._generate_plots(
            final_data,
            model_name,
            shots_selected,
            filtered_datasets,
            base_results_dir
        )

        choices_order_none_final_data = final_data[final_data['choices_order'] == "none"]
        #
        # -------------------------
        # 2) NEW: Analyze each column individually
        # -------------------------
        self._analyze_each_dimension(
            choices_order_none_final_data,  # or grouped_data, depending on your filtering needs
            model_name,
            shots_selected,
            filtered_datasets,
            base_results_dir
        )
        self.save_heatmap2(choices_order_none_final_data, model_name, shots_selected, filtered_datasets,
                           base_results_dir)
        self.save_heatmap_agg(final_data, model_name, shots_selected, filtered_datasets, base_results_dir)
        self.save_heatmap_agg_2(final_data, model_name, shots_selected, filtered_datasets, base_results_dir)
        total_time = time.time() - start_time
        print(f"Total processing time for {model_name}: {total_time:.2f} seconds")
        print("-" * 50)
        return filtered_datasets

    def _aggregate_configuration_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate scores for each configuration combination by grouping on:
        (dataset, template, separator, enumerator, choices_order).

        In this version, we assume the raw DataFrame has columns like
        'score' (rather than pre-aggregated 'sum_scores' and 'count').
        """
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
        """
        Filter configurations to those with exactly 100 examples,
        then select the top 5 datasets (by number of quads) and union them
        with the user-supplied 'interesting_datasets'.

        Returns the final set of dataset names to include in analysis.
        """
        # Filter configurations with exactly 100 examples
        filtered_df = data[data["count"] == 100].copy()

        # Get datasets with the most configurations
        dataset_counts = (
            filtered_df.groupby("dataset")["count"]
            .count()
            .reset_index(name="num_quads")
        )

        top5_datasets = dataset_counts.nlargest(5, "num_quads")["dataset"].tolist()
        return set(interesting_datasets).union(top5_datasets)

    def _prepare_visualization_data(
            self,
            data: pd.DataFrame,
            selected_datasets: Set[str]
    ) -> pd.DataFrame:
        """
        Prepare data for visualization by filtering to the selected datasets
        and creating a 'quad' column that concatenates
        (template, separator, enumerator, choices_order).
        """
        filtered_df = data[data["dataset"].isin(selected_datasets)].copy()

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
        """
        Generate and save visualization plots for the 'quad' analysis.
        Each dataset gets an HTML scatter plot with points labeled by 'count'.
        """
        # Prepare directory
        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )
        os.makedirs(model_dir, exist_ok=True)

        print(f"Generating and saving 'quad' plots for model: {model_name}")
        plot_start = time.time()

        # Generate plot for each dataset
        for dataset in datasets:
            subset = data[data["dataset"] == dataset]
            if subset.empty:
                continue

            # fig = self._create_accuracy_plot(subset, dataset, title_suffix="(Quad Analysis)")
            dataset_dir = os.path.join(model_dir, f"{dataset.replace('/', '_')}")
            os.makedirs(dataset_dir, exist_ok=True)

            # Save plot
            output_fig_file = os.path.join(dataset_dir, "prompt_configuration_analyzer.html")
            # fig.write_html(output_fig_file)

            # Save data as parquet
            output_parquet_file = os.path.join(dataset_dir, "prompt_configuration_analyzer.parquet")
            subset.to_parquet(output_parquet_file, index=False)

        print(f"Plotting completed in {time.time() - plot_start:.2f} seconds")

    def plot_accuracy_distribution_dim(self, config_data: pd.DataFrame, x_column="mean_accuracy"):
        """Create a bar chart of accuracy values with configuration counts."""
        grouped_data = config_data.groupby('quad').agg({
            x_column: 'mean',
            'combination_count': 'first'
        }).reset_index()

        fig = px.bar(
            grouped_data,
            x='quad',
            y=x_column,
            text='combination_count',
            title="Accuracy by Configuration Type"
        )

        fig.update_traces(
            textposition='outside',
            cliponaxis=False,
            textfont=dict(size=12, color='black'),
            texttemplate='Configurations: %{text:,}'
        )

        fig.update_layout(
            xaxis_title="Configuration Type",
            yaxis_title="Average Accuracy Score (0-1)",
            title={
                'text': "Accuracy by Configuration Type<br><sup>Each bar shows a configuration's accuracy with the number of variations tested</sup>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(t=50)
        )
        return fig

    def _create_accuracy_plot(
            self,
            data: pd.DataFrame,
            dataset: str,
            title_suffix: str = ""
    ):
        """
        Create an accuracy scatter plot for a specific dataset, labeling each point by its 'count'.
        """
        fig = px.scatter(
            data,
            x="quad",
            y="accuracy",
            text="count",
            title=f"Accuracy by Quad - {dataset} {title_suffix}",
            labels={"quad": "Configuration", "accuracy": "Accuracy"}
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(range=[0, 1.05])
        )
        return fig

    # -------------------------------------------------------------------------
    # NEW CODE: Analyzing each dimension separately (template, separator, etc.)
    # -------------------------------------------------------------------------

    def _analyze_each_dimension(
            self,
            data: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            datasets: Set[str],
            base_results_dir: str
    ) -> None:
        """
        For each of the four columns (template, separator, enumerator, choices_order),
        compute how that column performs on average (summing over all possible values
        of the other three columns).

        Then generate plots and save Parquet files in subdirectories of the base directory.
        """
        columns_to_analyze = ["template", "separator", "enumerator", "choices_order"]
        # 1) Compute dimension-level aggregates
        for dataset in datasets:
            for column in columns_to_analyze:
                subset = data[data["dataset"] == dataset]
                if subset.empty:
                    continue
                dim_df = self._compute_dimension_averages(subset, column)
                if dim_df.empty:
                    continue
                # 2) Generate dimension-based plots for each dataset
                self._generate_dimension_plots(
                    dim_df,
                    column,
                    model_name,
                    shots_selected,
                    dataset,
                    base_results_dir
                )

    def _compute_dimension_averages(
            self,
            data: pd.DataFrame,
            focus_column: str
    ) -> pd.DataFrame:
        """
        Given the full quad-based data, compute the average performance for 'focus_column'
        by grouping over [dataset, focus_column] and summing over all possible values
        in the other three columns.

        We also keep track of how many distinct combos (in the other three columns)
        contributed to each row, storing that in 'combination_count'.
        """
        # The four columns
        all_columns = ["template", "separator", "enumerator", "choices_order"]
        # The other three columns
        other_cols = [c for c in all_columns if c != focus_column]

        # Create an identifier for the other columns to count distinct combos
        # (since we want to know how many different ways the other 3 columns can appear).
        data = data.copy()  # Safe copy to avoid modifying original outside
        data["other_combo"] = data[other_cols].apply(lambda row: tuple(row), axis=1)

        # Group by dataset and the focus column
        grouped = data.groupby([focus_column])

        # Sum up sum_scores, sum up counts
        dim_acc_mean = grouped["accuracy"].mean().round(2)
        dim_acc_median = grouped["accuracy"].median().round(2)
        dim_total_count_mean = grouped["count"].mean().round(2)
        dim_total_count_median = grouped["count"].median().round(2)

        # Count how many distinct combos contributed
        dim_combo_count = grouped["other_combo"].nunique()

        # Build the dimension DataFrame
        dim_df = pd.DataFrame({
            "dataset": data.dataset.iloc[0],
            focus_column: [idx for idx in dim_acc_mean.index],
            "mean_accuracy": dim_acc_mean,
            "median_accuracy": dim_acc_median,
            "mean_total_count": dim_total_count_mean,
            "median_total_count": dim_total_count_median,
            "combination_count": dim_combo_count
        })

        # For consistency with the original structure, fill the other columns with "ALL"
        for col in other_cols:
            dim_df[col] = "ALL"
        # Now overwrite the focus column with its actual value from grouping

        # Create a 'quad' label for plotting, e.g., "X | ALL | ALL | ALL"
        dim_df["quad"] = (
                dim_df["template"] + " | " +
                dim_df["separator"] + " | " +
                dim_df["enumerator"] + " | " +
                dim_df["choices_order"]
        )

        # Clean up temporary column
        dim_df = dim_df.reset_index(drop=True)
        return dim_df

    def _generate_dimension_plots(
            self,
            dim_df_subset: pd.DataFrame,
            focus_column: str,
            model_name: str,
            shots_selected: int,
            dataset: str,
            base_results_dir: str
    ) -> None:
        """
        Generate and save dimension-based plots for a given focus column.

        For each dataset:
          - Create a subdirectory named after the focus column.
          - Build two scatter plots:
             1) Scatter of 'quad' vs. 'mean_accuracy'.
             2) Scatter of 'quad' vs. 'median_accuracy'.
          - Annotate each point with 'combination_count'.
          - Save the figure to HTML and the underlying data to Parquet.

        Args:
            dim_df: DataFrame returned by _compute_dimension_averages, containing:
                    ['dataset', focus_column, 'mean_accuracy', 'median_accuracy',
                     'mean_total_count', 'median_total_count', 'combination_count', 'quad', ...]
            focus_column: The column we are focusing on (e.g. "template", "separator", etc.)
            model_name: Name of the model being analyzed
            shots_selected: Number of shots used in the experiment
            datasets: Set of dataset names to include in analysis
            base_results_dir: Base directory where results should be saved
        """
        # Prepare the root directory for saving results for this model and shots
        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )
        os.makedirs(model_dir, exist_ok=True)

        print(f"Generating dimension-based plots for '{focus_column}'...")

        # We will plot mean_accuracy and median_accuracy
        metrics_to_plot = ["mean_accuracy", "median_accuracy"]

        # Create a dimension-specific subdirectory under the dataset folder
        dataset_dir = os.path.join(model_dir, f"{dataset.replace('/', '_')}")
        dimension_dir = os.path.join(dataset_dir, focus_column)
        os.makedirs(dimension_dir, exist_ok=True)
        # Also save the entire dimension DataFrame subset to Parquet
        output_parquet_file = os.path.join(dimension_dir, f"prompt_configuration_analyzer.parquet")
        dim_df_subset.to_parquet(output_parquet_file, index=False)

        # Generate and save a plot for each metric
        # for metric in metrics_to_plot:
        #     fig = px.scatter(
        #         dim_df_subset,
        #         x="quad",
        #         y=metric,
        #         text="combination_count",
        #         title=f"{metric.replace('_', ' ').title()} by '{focus_column}' - {dataset}",
        #         labels={
        #             "quad": f"{focus_column} (others = ALL)",
        #             metric: metric.replace("_", " ").title()
        #         }
        #     )
        #     fig.update_traces(textposition="top center")
        #     fig.update_layout(
        #         xaxis=dict(tickangle=45),
        #         yaxis=dict(range=[0, 1.05]),
        #     )

        # Save the figure as an HTML file
        # output_fig_file = os.path.join(dimension_dir, f"prompt_configuration_analyzer.html")
        # fig.write_html(output_fig_file)

    def save_heatmap(self, final_data, model_name, shots_selected, filtered_datasets, base_results_dir):

        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )

        final_data = final_data.drop(columns='choices_order')

        for dataset in filtered_datasets:
            subset = final_data[final_data["dataset"] == dataset]

            heatmap_dir = os.path.join(model_dir, "2D_Heatmaps", dataset.replace('/', '_'))
            os.makedirs(heatmap_dir, exist_ok=True)

            for template in subset['template'].unique():
                template_data = subset[subset['template'] == template]

                pivot_table = pd.pivot_table(
                    template_data,
                    values='accuracy',
                    index='separator',
                    columns='enumerator',
                    aggfunc='mean'
                )

                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_table,
                            annot=True,
                            fmt='.3f',
                            cmap='viridis',
                            cbar_kws={'label': 'Accuracy'})

                plt.title(
                    f'2D Heatmap for {dataset}\nTemplate: {template}\n(Model: {model_name}, Shots: {shots_selected})')
                plt.xlabel('Enumerator')
                plt.ylabel('Separator')

                output_fig_file = os.path.join(heatmap_dir, f"heatmap_template_{template.replace('/', '_')}.png")
                plt.savefig(output_fig_file, dpi=100, bbox_inches='tight')
                plt.close()

                if not pivot_table.isna().any().any():

                    col_best_dir = os.path.join(model_dir, "2D_Heatmaps_special", "col_best")
                    col_worst_dir = os.path.join(model_dir, "2D_Heatmaps_special", "col_worst")
                    row_best_dir = os.path.join(model_dir, "2D_Heatmaps_special", "row_best")
                    row_worst_dir = os.path.join(model_dir, "2D_Heatmaps_special", "row_worst")

                    os.makedirs(col_best_dir, exist_ok=True)
                    os.makedirs(col_worst_dir, exist_ok=True)
                    os.makedirs(row_best_dir, exist_ok=True)
                    os.makedirs(row_worst_dir, exist_ok=True)

                    def plot_and_save_special(output_dir, prefix):

                        plt.figure(figsize=(10, 8))
                        sns.heatmap(
                            pivot_table,
                            annot=True,
                            fmt='.3f',
                            cmap='viridis',
                            cbar_kws={'label': 'Accuracy'}
                        )
                        plt.title(
                            f'2D Heatmap for {dataset}\n'
                            f'Template: {template}\n'
                            f'(Model: {model_name}, Shots: {shots_selected})'
                        )
                        plt.xlabel('Enumerator')
                        plt.ylabel('Separator')

                        special_filename = (
                            f"{prefix}_{dataset.replace('/', '_')}_"
                            f"heatmap_template_{template.replace('/', '_')}.png"
                        )
                        output_special_file = os.path.join(output_dir, special_filename)
                        plt.savefig(output_special_file, dpi=100, bbox_inches='tight')
                        plt.close()

                    col_max = pivot_table.idxmax(axis=1)
                    if col_max.nunique() == 1:
                        best_col = col_max.iloc[0]
                        plot_and_save_special(col_best_dir, best_col)

                    col_min = pivot_table.idxmin(axis=1)
                    if col_min.nunique() == 1:
                        worst_col = col_min.iloc[0]
                        plot_and_save_special(col_worst_dir, worst_col)

                    row_max = pivot_table.idxmax(axis=0)
                    if row_max.nunique() == 1:
                        best_row = row_max.iloc[0]
                        plot_and_save_special(row_best_dir, best_row)

                    row_min = pivot_table.idxmin(axis=0)
                    if row_min.nunique() == 1:
                        worst_row = row_min.iloc[0]
                        plot_and_save_special(row_worst_dir, worst_row)

    def save_heatmap_agg(self, final_data, model_name, shots_selected, filtered_datasets, base_results_dir):

        final_data = final_data.groupby(['dataset', 'template', 'separator', 'enumerator']).agg(
            accuracy=pd.NamedAgg(column='accuracy', aggfunc='median')).reset_index()

        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )

        for dataset in filtered_datasets:
            subset = final_data[final_data["dataset"] == dataset]

            heatmap_dir = os.path.join(model_dir, "2D_Heatmaps_Agg", dataset.replace('/', '_'))
            os.makedirs(heatmap_dir, exist_ok=True)

            for template in subset['template'].unique():
                template_data = subset[subset['template'] == template]

                pivot_table = pd.pivot_table(
                    template_data,
                    values='accuracy',
                    index='separator',
                    columns='enumerator',
                    aggfunc='mean'
                )

                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_table,
                            annot=True,
                            fmt='.3f',
                            cmap='viridis',
                            cbar_kws={'label': 'Accuracy'})

                plt.title(
                    f'2D Heatmap for {dataset}\nTemplate: {template}\n(Model: {model_name}, Shots: {shots_selected})')
                plt.xlabel('Enumerator')
                plt.ylabel('Separator')

                output_fig_file = os.path.join(heatmap_dir, f"heatmap_template_{template.replace('/', '_')}.png")
                plt.savefig(output_fig_file, dpi=100, bbox_inches='tight')
                plt.close()

                if not pivot_table.isna().any().any():

                    col_best_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg", "col_best")
                    col_worst_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg", "col_worst")
                    row_best_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg", "row_best")
                    row_worst_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg", "row_worst")

                    os.makedirs(col_best_dir, exist_ok=True)
                    os.makedirs(col_worst_dir, exist_ok=True)
                    os.makedirs(row_best_dir, exist_ok=True)
                    os.makedirs(row_worst_dir, exist_ok=True)

                    def plot_and_save_special(output_dir, prefix):

                        plt.figure(figsize=(10, 8))
                        sns.heatmap(
                            pivot_table,
                            annot=True,
                            fmt='.3f',
                            cmap='viridis',
                            cbar_kws={'label': 'Accuracy'}
                        )
                        plt.title(
                            f'2D Heatmap for {dataset}\n'
                            f'Template: {template}\n'
                            f'(Model: {model_name}, Shots: {shots_selected})'
                        )
                        plt.xlabel('Enumerator')
                        plt.ylabel('Separator')

                        special_filename = (
                            f"{prefix}_{dataset.replace('/', '_')}_"
                            f"heatmap_template_{template.replace('/', '_')}.png"
                        )
                        output_special_file = os.path.join(output_dir, special_filename)
                        plt.savefig(output_special_file, dpi=100, bbox_inches='tight')
                        plt.close()

                    col_max = pivot_table.idxmax(axis=1)
                    if col_max.nunique() == 1:
                        best_col = col_max.iloc[0]
                        plot_and_save_special(col_best_dir, best_col)

                    col_min = pivot_table.idxmin(axis=1)
                    if col_min.nunique() == 1:
                        worst_col = col_min.iloc[0]
                        plot_and_save_special(col_worst_dir, worst_col)

                    row_max = pivot_table.idxmax(axis=0)
                    if row_max.nunique() == 1:
                        best_row = row_max.iloc[0]
                        plot_and_save_special(row_best_dir, best_row)

                    row_min = pivot_table.idxmin(axis=0)
                    if row_min.nunique() == 1:
                        worst_row = row_min.iloc[0]
                        plot_and_save_special(row_worst_dir, worst_row)

    def save_heatmap_agg_2(self, final_data, model_name, shots_selected, filtered_datasets, base_results_dir):

        final_data = final_data.groupby(['dataset', 'separator', 'enumerator']).agg(
            accuracy=pd.NamedAgg(column='accuracy', aggfunc='median')).reset_index()

        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )

        for dataset in filtered_datasets:
            subset = final_data[final_data["dataset"] == dataset]

            heatmap_dir = os.path.join(model_dir, "2D_Heatmaps_Agg_with_temp", dataset.replace('/', '_'))
            os.makedirs(heatmap_dir, exist_ok=True)

            pivot_table = pd.pivot_table(
                subset,
                values='accuracy',
                index='separator',
                columns='enumerator',
                aggfunc='mean'
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table,
                        annot=True,
                        fmt='.3f',
                        cmap='viridis',
                        cbar_kws={'label': 'Accuracy'})

            plt.title(
                f'2D Heatmap for {dataset}\nAgg Templates \n(Model: {model_name}, Shots: {shots_selected})')
            plt.xlabel('Enumerator')
            plt.ylabel('Separator')

            output_fig_file = os.path.join(heatmap_dir, f"heatmap_agg_templates.png")
            plt.savefig(output_fig_file, dpi=100, bbox_inches='tight')
            plt.close()

            if not pivot_table.isna().any().any():

                col_best_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg_with_temp", "col_best")
                col_worst_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg_with_temp", "col_worst")
                row_best_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg_with_temp", "row_best")
                row_worst_dir = os.path.join(model_dir, "2D_Heatmaps_special_Agg_with_temp", "row_worst")

                os.makedirs(col_best_dir, exist_ok=True)
                os.makedirs(col_worst_dir, exist_ok=True)
                os.makedirs(row_best_dir, exist_ok=True)
                os.makedirs(row_worst_dir, exist_ok=True)

                def plot_and_save_special(output_dir, prefix):
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        pivot_table,
                        annot=True,
                        fmt='.3f',
                        cmap='viridis',
                        cbar_kws={'label': 'Accuracy'}
                    )
                    plt.title(
                        f'2D Heatmap for {dataset}\n'
                        f'Agg Templates\n'
                        f'(Model: {model_name}, Shots: {shots_selected})'
                    )
                    plt.xlabel('Enumerator')
                    plt.ylabel('Separator')

                    # שם הקובץ: <prefix>_<dataset>_heatmap_template_<template>.png
                    # לדוגמה: lowercase_mmlu.global_facts_heatmap_template_MyTemplate.png
                    special_filename = (
                        f"{prefix}_{dataset.replace('/', '_')}_"
                        f"heatmap_templates_agg.png"
                    )
                    output_special_file = os.path.join(output_dir, special_filename)
                    plt.savefig(output_special_file, dpi=100, bbox_inches='tight')
                    plt.close()

                col_max = pivot_table.idxmax(axis=1)
                if col_max.nunique() == 1:
                    best_col = col_max.iloc[0]
                    plot_and_save_special(col_best_dir, best_col)

                col_min = pivot_table.idxmin(axis=1)
                if col_min.nunique() == 1:
                    worst_col = col_min.iloc[0]
                    plot_and_save_special(col_worst_dir, worst_col)

                row_max = pivot_table.idxmax(axis=0)
                if row_max.nunique() == 1:
                    best_row = row_max.iloc[0]
                    plot_and_save_special(row_best_dir, best_row)

                row_min = pivot_table.idxmin(axis=0)
                if row_min.nunique() == 1:
                    worst_row = row_min.iloc[0]
                    plot_and_save_special(row_worst_dir, worst_row)
