import os
import time
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Union

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
            self._process_accuracy_stats(dataset_df, model_dir, df_filtered)
            # Process and save enumerator analysis
            self._process_enumerator_analysis(
                dataset_df,
                model_name,
                dataset,
                model_dir
            )

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

    def _process_enumerator_analysis(
            self,
            df: pd.DataFrame,
            model_name: str,
            dataset: str,
            model_dir: str,
            threshold: float = 0.1
    ) -> None:
        """Analyze answer distributions for poorly performing questions."""
        enum_dir = os.path.join(model_dir, dataset.replace('/', '_'), 'enumerator_analysis')
        os.makedirs(enum_dir, exist_ok=True)

        # Position mappings
        position_mappings = {
            'greek': "αβγδεζηθικ",
            'keyboard': "!@#$%^₪*)(",
            'capitals': "ABCDEFGHIJ",
            'lowercase': "abcdefghij",
            'numbers': [str(i + 1) for i in range(10)],
            'roman': ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        }

        def map_to_position(answer: str, enum_type: str) -> int:
            prefix = answer.split('.')[0].strip()
            mapping = position_mappings.get(enum_type, "")
            return mapping.index(prefix) + 1 if prefix in mapping else 0

        # Calculate question accuracies
        accuracy_stats = (
            df.groupby('sample_index')
            .agg({
                'score': ['sum', 'count']
            })
            .reset_index()
        )
        accuracy_stats.columns = ['sample_index', 'correct_count', 'total_count']
        accuracy_stats['accuracy'] = (accuracy_stats['correct_count'] / accuracy_stats['total_count']).round(3)

        # Identify poor performers
        poor_performers = accuracy_stats[accuracy_stats['accuracy'] < threshold]

        if not poor_performers.empty:
            # Create main scatter plot for most common answers
            position_stats = self._create_position_stats(
                df,
                poor_performers['sample_index'],
                map_to_position,
                accuracy_stats
            )

            # Save position statistics
            position_stats.to_parquet(os.path.join(enum_dir, 'position_distribution.parquet'))

            # Create and save scatter plot
            self._create_scatter_plot(
                position_stats,
                model_name,
                dataset,
                threshold,
                enum_dir
            )

            # Create and save heatmap
            self._create_answer_heatmap(
                df,
                poor_performers['sample_index'],
                map_to_position,
                enum_dir,
                model_name,
                dataset
            )

    def _create_answer_heatmap(
            self,
            df: pd.DataFrame,
            poor_performers_idx: pd.Index,
            position_mapping_func,
            save_dir: str,
            model_name: str,
            dataset: str
    ) -> None:
        """Create a normalized confusion matrix of answer distributions.

        Each cell shows the proportion of answers for that position (0 to 1),
        with rows summing to 1 across all possible answer positions.
        """
        # Get answer positions and counts
        answer_distributions = (
            df[df['sample_index'].isin(poor_performers_idx)]
            .assign(
                position=lambda x: x.apply(
                    lambda row: position_mapping_func(row['closest_answer'], row['enumerator']),
                    axis=1
                )
            )
            .groupby(['sample_index', 'position'])
            .size()
            .reset_index(name='count')
        )

        # Calculate proportions for each question
        answer_distributions['total'] = answer_distributions.groupby('sample_index')['count'].transform('sum')
        answer_distributions['proportion'] = answer_distributions['count'] / answer_distributions['total']

        # Create the confusion matrix
        confusion_matrix = (
            answer_distributions
            .pivot(
                index='sample_index',
                columns='position',
                values='proportion'
            )
            .fillna(0)
        )

        # Sort by question index for consistent display
        confusion_matrix = confusion_matrix.sort_index()

        # Create heatmap
        fig = px.imshow(
            confusion_matrix,
            labels={
                'x': 'Answer Position',
                'y': 'Question ID',
                'color': 'Proportion'
            },
            x=[str(i) for i in confusion_matrix.columns],
            y=[str(i) for i in confusion_matrix.index],
            aspect='auto',
            title=f"{model_name} - {dataset}"
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Answer Position",
            yaxis_title="Question ID",
            coloraxis=dict(
                colorscale='Blues',
                cmin=0,
                cmax=1,
                colorbar_title="Proportion"
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Add informative hover text
        fig.update_traces(
            hovertemplate=(
                    "Question ID: %{y}<br>" +
                    "Answer Position: %{x}<br>" +
                    "Proportion: %{z:.3f}<br>" +
                    "<extra></extra>"
            )
        )

        # Save and display
        output_path = os.path.join(save_dir, 'answer_distribution_heatmap.html')
        fig.write_html(output_path)
        fig.show()
    def _create_position_stats(
            self,
            df: pd.DataFrame,
            poor_performers_idx: pd.Index,
            position_mapping_func,
            accuracy_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Create position statistics for scatter plot."""
        return (
            df[df['sample_index'].isin(poor_performers_idx)]
            .assign(
                position=lambda x: x.apply(
                    lambda row: position_mapping_func(row['closest_answer'], row['enumerator']),
                    axis=1
                )
            )
            .groupby(['sample_index', 'position'])
            .size()
            .reset_index(name='frequency')
            .merge(
                accuracy_stats[['sample_index', 'accuracy', 'correct_count', 'total_count']],
                on='sample_index'
            )
            .assign(
                percentage=lambda x: x.groupby('sample_index')['frequency']
                .transform(lambda y: y / y.sum() * 100)
            )
            .sort_values(['sample_index', 'frequency'], ascending=[True, False])
            .drop_duplicates('sample_index')
            .assign(question_number=lambda x: x['sample_index'])
            .round({'percentage': 2, 'accuracy': 3})
        )

    def _create_scatter_plot(
            self,
            position_stats: pd.DataFrame,
            model_name: str,
            dataset: str,
            threshold: float,
            save_dir: str
    ) -> None:
        """Create scatter plot of most common answer positions."""
        fig = px.scatter(
            position_stats,
            x='question_number',
            y='percentage',
            custom_data=['position', 'accuracy', 'correct_count', 'total_count'],
            title=f"{model_name} - {dataset}\nMost Common Answer Position (Accuracy < {threshold})",
            labels={
                'question_number': 'Question Number',
                'percentage': 'Percentage (%)'
            }
        )

        fig.update_traces(
            marker=dict(size=10),
            hovertemplate=(
                "Question: %{x}<br>"
                "Position: %{customdata[0]}<br>"
                "Frequency: %{y:.1f}%<br>"
                "Accuracy: %{customdata[1]:.3f}<br>"
                "Correct: %{customdata[2]} / %{customdata[3]}"
            )
        )

        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1,
                gridcolor='lightgrey',
                showgrid=True
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor='lightgrey',
                showgrid=True
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.show()
        fig.write_html(os.path.join(save_dir, 'position_distribution.html'))

    def _process_accuracy_stats(
            self,
            dataset_df: pd.DataFrame,
            model_dir: str,
            df_filtered: pd.DataFrame
    ) -> None:
        """Process accuracy statistics for a dataset."""
        overall_question_stats = self._aggregate_samples_accuracy(dataset_df)
        self._save_samples_accuracy_tables(overall_question_stats, model_dir, suffix="overall")

        dimensions = ["template", "separator", "enumerator", "choices_order"]
        for dim in dimensions:
            print(f"Computing question accuracy grouped by '{dim}'...")
            question_stats = self._aggregate_samples_accuracy_by_axis(df_filtered, dim)
            self._save_samples_accuracy_tables(question_stats, model_dir, dim, suffix=f"by_{dim}")

    def _prepare_output_directory(
            self,
            base_results_dir: str,
            shots_selected: int,
            model_name: str
    ) -> str:
        """Prepare output directory structure."""
        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
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
            suffix: str = "overall"
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

        # fig_file = os.path.join(plot_dir, f"fraction_correct_histogram_{suffix}.html")
        # fig.write_html(fig_file)
