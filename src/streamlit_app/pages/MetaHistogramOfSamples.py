import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.streamlit_app.ui_components.SamplesNavigator import SamplesNavigator
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
DatasetsConstants = Constants.DatasetsConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class MetaHistogramOfSamples:
    def __init__(self, dataset_sizes_path):
        self.dataset_sizes_path = dataset_sizes_path

    def display_page(self):
        st.title("Histogram of Samples")
        selected_results_file, model_files, shot = self.get_files()
        self.display_aggregated_results(
            selected_results_file, model_files, shot)

    def get_model_files(self, selected_results_file):
        folders = [file for file in selected_results_file.iterdir()
                   if file.is_dir()]
        models_names = {f.name: f for f in folders}
        sorted_folders = dict(
            sorted(models_names.items(), key=lambda x: (x[0].lower(), x[0]), reverse=False))
        models = st.sidebar.multiselect("Select models to visualize", list(sorted_folders.keys()),
                                        default=list(sorted_folders.keys()))
        # add the model to st.session_state
        st.session_state["models"] = models
        selected_models_files = [models_names[model] for model in models]
        return selected_models_files

    def get_files(self):
        main_results_path = ExperimentConstants.MAIN_RESULTS_PATH
        selected_results_file = ResultsLoader.get_folder_selections_options(
            main_results_path, "Select results folder to visualize", reverse=True
        )
        self.results_folder = main_results_path / selected_results_file
        shot = st.sidebar.selectbox("Select number of shots", [
                                    "zero_shot", "three_shot"])
        model_files = self.get_model_files(selected_results_file)
        return selected_results_file, model_files, shot

    def display_aggregated_results(self, selected_results_file, selected_models_files, shot):
        datasets_names = self.get_dataset_split_options()
        total_merge_df = MetaHistogramCalculator.aggregate_data_across_models(selected_results_file,
                                                                              selected_models_files, shot,
                                                                              datasets_names)
        total_merge_df = MetaHistogramCalculator.calculate_and_add_accuracy_columns(
            total_merge_df)
        self.plot_aggregated_histogram(total_merge_df)
        self.display_examples(total_merge_df)

    def get_dataset_split_options(self):
        other_datasets = DatasetsConstants.OTHER
        data_type = st.selectbox("Select datasets", [DatasetsConstants.MMLU_NAME,
                                                     DatasetsConstants.MMLU_PRO_NAME] + other_datasets)
        ds_size = TemplatesGeneratorConstants.DATASET_SIZES_PATH
        df_ds_size = pd.read_csv(ds_size)
        if data_type == DatasetsConstants.MMLU_NAME:
            df_ds_size = df_ds_size[
                ~df_ds_size['Name'].str.startswith(
                    tuple(d for d in [DatasetsConstants.MMLU_NAME, DatasetsConstants.MMLU_PRO_NAME]
                          if d != data_type)
                )]
        df_ds_size = df_ds_size[df_ds_size["Name"].str.startswith(data_type)]
        datasets_names = df_ds_size["Name"].tolist()
        # datasets_names = MMLUSplitter.get_data_files(MMLUConstants.ALL_NAMES,
        #                                              MMLUSplitter.get_data_options(MMLUConstants.ALL_NAMES))
        return datasets_names

    def plot_aggregated_histogram(self, df):
        """
        Plot the histogram of the results.
        @param df: DataFrame containing the data to plot.
        """
        # Get the mean of number_of_predictions
        mean_predictions = df['number_of_predictions'].mean()

        # Check if mean is NaN
        if np.isnan(mean_predictions):
            config_count = 0  # or any default value you want to use
        else:
            config_count = int(mean_predictions)

        title = f"Aggregated Histogram by"
        st.markdown(
            f"There are {len(df)} examples in the dataset across {config_count} configurations "
        )
        st.markdown(
            f"The models in the plots are {', '.join(st.session_state['models'])}")
        # Setting up the plot
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.grid(True)  # Add grid lines for better readability
        # Ensure grid lines are behind other plot elements
        ax.set_axisbelow(True)

        # Plotting the histogram
        # Adjust bins to include the range from 12 to 100
        bins = np.arange(0, 105, 5)
        df['accuracy'].plot(kind='hist', bins=bins, ax=ax,
                            color='skyblue', edgecolor='black')

        # Adding title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(
            "Percentage of Templates with Correct Predictions", fontsize=14)
        ax.set_ylabel("Number of Examples", fontsize=14)

        # Customizing tick labels from and plot all the bins
        ax.set_xticks(bins)
        ax.set_xticklabels([f'{i}' for i in bins])
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add the number of examples in the histogram bars in the center
        for rect in ax.patches:
            height = rect.get_height()
            if height:
                ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12, color='blue', fontweight='bold')

        # Display the plot
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        st.pyplot(fig)

    def display_examples(self, df):
        min_percentage = st.slider("Minimum percentage of examples to display", 0, 100, 0, step=5)
        max_percentage = st.slider("Maximum percentage of examples to display", 0, 100, 5, step=5)
        examples = df[(df['accuracy'] > min_percentage) & (df['accuracy'] < max_percentage)]
        self.display_example_details(examples)
        # Add categorization section after the regular display
        st.markdown("---")
        st.markdown("## Error Category Analysis")
        example_data = MetaHistogramCalculator.extract_example_data(examples)
        self.display_error_categorization(example_data)

    def display_example_details(self, examples):
        # Implementation to display details for each example
        example_data = MetaHistogramCalculator.extract_example_data(examples)
        self.display_bar_chart(example_data)
        self.display_bar_chart_precentages(example_data)  # Precentage Chart
        self.display_text_examples(example_data)

    def display_bar_chart(self, example_data):
        fig = px.bar(example_data['dataset'].value_counts(), x=example_data['dataset'].value_counts().index,
                     y=example_data['dataset'].value_counts().values,
                     labels={'x': 'dataset', 'y': 'number of examples'},
                     title='Number of examples for each dataset in the selected range')
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    def get_number_of_examples_per_topic(self):
        df = pd.read_csv(self.dataset_sizes_path)
        return dict(zip(df['Name'], df['test']))

    def display_bar_chart_precentages(self, example_data):
        try:
            dataset_size = self.get_number_of_examples_per_topic()
            current_count = example_data['dataset'].value_counts()

            percentages = {
                dataset: (current_count[dataset] / dataset_size[dataset] * 100)
                for dataset in current_count.index
                if dataset in dataset_size
            }

            percentage_series = pd.Series(
                percentages).sort_values(ascending=False)

            fig = px.bar(
                x=percentage_series.index,
                y=percentage_series.values,
                labels={'x': 'Dataset', 'y': 'Percentage of Examples'},
                title='Percentage of examples in the selected range from total e',
            )

            fig.update_traces(texttemplate='%{y:.0f}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")

    def display_text_examples(self, example_data):
        dataset = st.selectbox(
            "Select dataset", example_data['dataset'].unique())
        # results_folder = ResultConstants.MAIN_RESULTS_PATH / Path("MultipleChoiceTemplatesStructured")
        default_model = "Mistral-7B-Instruct-v0.2"
        models = sorted(
            [model for model in os.listdir(self.results_folder) if os.path.isdir(self.results_folder / model)])
        model = st.selectbox("Select model", models,
                             index=models.index(default_model) if default_model in models else 0)
        selected_examples = example_data[example_data['dataset'] == dataset]
        # current_instance = selected_examples.iloc[st.session_state.get('file_index', 0)]
        current_instance = self.get_current_index(selected_examples, dataset)
        full_results_path = self.results_folder / model / dataset / \
            "zero_shot" / "empty_system_format" / "experiment_template_0.json"
        with open(full_results_path, "r") as file:
            template = json.load(file)
        sample = template["results"]["test"][current_instance]
        self.display_sample_details(sample)
        st.write("----")

    def get_current_index(self, examples_id, dataset):
        dataset_index = examples_id[examples_id['dataset']
                                    == dataset]['example_number'].values

        # sort the files by the number of the experiment
        # write on the center of the page
        st.markdown(f"#### {len(dataset_index)} Examples",
                    unsafe_allow_html=True)
        if "file_index" not in st.session_state:
            st.session_state["file_index"] = 0
        if "dataset" not in st.session_state:
            st.session_state["dataset"] = dataset
        if dataset != st.session_state["dataset"]:
            st.session_state["file_index"] = 0
            st.session_state["dataset"] = dataset

        st.session_state["files_number"] = len(dataset_index)

        # add bottoms to choose example
        col1, col2 = st.columns(2)
        with col1:
            st.button(label="Previous sentence",
                      on_click=SamplesNavigator.previous_sentence)
        with col2:
            st.button(label="Next sentence",
                      on_click=SamplesNavigator.next_sentence)
        st.selectbox(
            "Sentences",
            [f"sentence {i + 1}" for i in range(0,
                                                st.session_state["files_number"])],
            index=st.session_state["file_index"],
            on_change=SamplesNavigator.go_to_sentence,
            key="selected_sentence",
        )

        current_instance = examples_id[examples_id['example_number']
                                       == dataset_index[st.session_state['file_index']]]
        current_instance = int(current_instance.example_number.values[0])
        return current_instance

    def display_sample_details(self, sample):
        formatted_instance = sample['Instance'].replace(
            '\n\n', '<br><br>').replace('\n', '<br>')
        formatted_ground_truth = sample['GroundTruth']
        formatted_prediction = sample['Result']
        formatted_score = sample['Score']

        st.markdown(
            f"**Instance**: {formatted_instance}<br>"
            f"**Ground Truth**: {formatted_ground_truth}<br>"
            f"**Predicted**: {formatted_prediction}<br>"
            f"**Score**: {formatted_score}",
            unsafe_allow_html=True
        )

    def display_error_categorization(self, examples):
        # Initialize annotation manager
        if 'annotation_manager' not in st.session_state:
            st.session_state.annotation_manager = AnnotationManager()
        try:
            annotations_file = Path(Constants.ResultConstants.ANNOTATIONS_FILE)
            if st.button("🗑️ Clear All Annotations"):
                if annotations_file.exists():
                    with open(annotations_file, 'w') as f:
                        json.dump({"annotations": [], "total_annotations": 0}, f, indent=2)
                    st.success("All annotations cleared successfully!")
                    st.experimental_rerun()

            mistake_examples = pd.DataFrame()
            default_model = Constants.LLMProcessorConstants.MISTRAL_V2_MODEL
            model = st.session_state.get('models', [default_model])[0]

            for idx, row in examples.iterrows():
                try:
                    sample = self._get_example_details(row, model)
                    if sample and sample.get('Score', 1) == 0:
                        mistake_examples = pd.concat([mistake_examples, pd.DataFrame([row])])
                except Exception:
                    continue

            if len(mistake_examples) == 0:
                st.warning("No mistakes found in the current selection.")
                return

            max_examples = st.number_input(
                "Number of examples to analyze:",
                min_value=1,
                max_value=min(len(mistake_examples), 100),
                value=min(10, len(mistake_examples))
            )

            selected_examples = self._sample_examples(mistake_examples, max_examples)

            # Store categories in session state
            if 'example_categories' not in st.session_state:
                st.session_state.example_categories = {}

            self._display_error_categories_legend()

            # Create form for all examples
            with st.form("annotation_form"):
                for i, (idx, row) in enumerate(selected_examples.iterrows()):
                    st.markdown("---")
                    with st.expander(f"Example {i + 1} (Dataset: {row['dataset']})", expanded=True):
                        try:
                            sample = self._get_example_details(row, model)
                            if sample:
                                self._display_example_content(sample)
                                category = st.selectbox(
                                    "Select error category:",
                                    ErrorCategories.get_all_categories(),
                                    key=f"cat_{idx}"
                                )
                                st.session_state.example_categories[idx] = {
                                    "category": category,
                                    "sample": sample,
                                    "dataset": row['dataset']
                                }
                        except Exception as e:
                            st.error(f"Error loading example {i + 1}: {str(e)}")

                if st.form_submit_button("Save All Annotations"):
                    if annotations_file.exists():
                        with open(annotations_file, 'r') as f:
                            all_annotations = json.load(f)
                    else:
                        all_annotations = {"annotations": [], "total_annotations": 0}

                    # Add new annotations
                    for idx, data in st.session_state.example_categories.items():
                        annotation_data = {
                            "model": str(model),
                            "id": str(idx),
                            "dataset": data["dataset"],
                            "category": data["category"],
                            "example_content": {
                                "instance": data["sample"]['Instance'],
                                "ground_truth": data["sample"]['GroundTruth'],
                                "prediction": data["sample"]['Result'],
                            }
                        }
                        all_annotations["annotations"].append(annotation_data)

                    all_annotations["total_annotations"] = len(all_annotations["annotations"])

                    with open(annotations_file, 'w') as f:
                        json.dump(all_annotations, f, indent=2)
                    st.success("All annotations saved successfully!")

            st.markdown("### Annotation Statistics")
            if annotations_file.exists():
                with open(annotations_file, 'r') as f:
                    current_annotations = json.load(f)

                if current_annotations["total_annotations"] > 0:
                    model_annotations = [ann for ann in current_annotations["annotations"]
                                         if ann.get('model') == model]

                    if model_annotations:
                        categories = [ann['category'] for ann in model_annotations]
                        category_counts = pd.Series(categories).value_counts()

                        fig = px.bar(
                            x=category_counts.index,
                            y=category_counts.values,
                            labels={'x': 'Category', 'y': 'Count'},
                            title=f'Distribution of Error Categories for {model}'
                        )
                        fig.update_traces(texttemplate='%{y}', textposition='outside')
                        st.plotly_chart(fig)

                        datasets = [ann['dataset'] for ann in model_annotations]
                        dataset_counts = pd.Series(datasets).value_counts()

                        fig = px.bar(
                            x=dataset_counts.index,
                            y=dataset_counts.values,
                            labels={'x': 'Dataset', 'y': 'Count'},
                            title=f'Distribution of Annotations by Dataset for {model}'
                        )
                        fig.update_traces(texttemplate='%{y}', textposition='outside')
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)

                        st.info(f"Total annotations for {model}: {len(model_annotations)}")
                    else:
                        st.info(f"No annotations yet for {model}")
                else:
                    st.info("No annotations have been made yet. Start by selecting categories for the examples above.")

        except Exception as e:
            st.error(f"Error in error categorization display: {str(e)}")

    def _sample_examples(self, examples, max_examples):
        """Sample examples from 3 datasets with most mistakes."""
        top_3_datasets = examples['dataset'].value_counts().head(3).index
        samples_per_dataset = max_examples // 3
        remainder = max_examples % 3
        sampled = []
        for i, dataset in enumerate(top_3_datasets):
            # Add one extra sample for any remainder
            current_samples = samples_per_dataset + (1 if i < remainder else 0)
            dataset_examples = examples[examples['dataset'] == dataset].head(current_samples)
            sampled.append(dataset_examples)
        return pd.concat(sampled)

    def _display_error_categories_legend(self):
        st.markdown("""
        #### Error Categories
        1. **Format Error** - Model fails to follow format or provide answer

        2. **Wrong Reasoning** - Model's logic path contains flaws

        3. **Wrong Annotation** - Answer could be debatable or incorrect

        4. **Lack of Knowledge** - Model lacks necessary knowledge that's not in prompt
        """)

    def _display_single_example_annotation(self, index, idx, row):
        st.markdown("---")
        with st.expander(f"Example {index + 1} (Dataset: {row['dataset']})", expanded=True):
            try:
                default_model = Constants.LLMProcessorConstants.MISTRAL_V2_MODEL
                model = st.session_state.get('models', [default_model])[0]
                sample = self._get_example_details(row, model)

                if sample:
                    self._display_example_content(sample)
                    self._add_example_annotation(idx, row['dataset'])
            except Exception as e:
                st.error(f"Error loading example {index + 1}: {str(e)}")

    def _get_example_details(self, row, model):
        try:
            full_results_path = (
                    self.results_folder /
                    model /
                    row['dataset'] /
                    Constants.ResultConstants.ZERO_SHOT /
                    Constants.ResultConstants.EMPTY_SYSTEM_FORMAT /
                    "experiment_template_0.json"
            )
            
            with open(full_results_path, "r") as file:
                template = json.load(file)
            return template["results"]["test"][int(row['example_number'])]
        except Exception as e:
            st.error(f"Error loading example details: {str(e)}")
            return None

    def _display_example_content(self, sample):
        """Display example content in a consistent format."""
        st.markdown("#### Example Content")
        st.markdown(f"**Question**:\n{sample['Instance']}")
        st.markdown(f"**Ground Truth**:\n{sample['GroundTruth']}")
        st.markdown(f"**Predicted**:\n{sample['Result']}")
        st.markdown(f"**Score**: {sample['Score']}")

    def _add_example_annotation(self, idx, dataset):
        """Add annotation for a single example."""
        category = st.selectbox(
            "Select error category:",
            ErrorCategories.get_all_categories(),
            key=f"cat_{idx}"
        )
        
        st.session_state.current_session_annotations[idx] = {
            'category': category,
            'dataset': dataset
        }

    def _handle_clear_annotations(self):
        st.session_state.all_annotations = {}
        st.session_state.current_session_annotations = {}
        st.success("All annotations have been cleared!")
        st.experimental_rerun()


    def display_annotation_statistics(self, annotations):
        if not annotations:
            st.warning("No annotations available to display statistics.")
            return
            
        st.markdown("### Annotation Statistics")
        categories = [ann['category'] for ann in annotations.values()]
        datasets = [ann['dataset'] for ann in annotations.values()]

        # Display only category distribution and dataset distribution
        tab1 = st.tabs(["Category Distribution"])[0]

        with tab1:
            self._plot_category_distribution(categories)

        #  dataset distribution but kept for future use
        # with tab2:
        #     self._plot_dataset_distribution(datasets)

    def _plot_category_distribution(self, categories):
        """Plot distribution of error categories."""
        category_counts = pd.Series(categories).value_counts()
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={'x': 'Category', 'y': 'Count'},
            title='Distribution of Error Categories'
        )
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig)

    def _plot_dataset_distribution(self, datasets):
        """Plot distribution of errors by dataset."""
        dataset_counts = pd.Series(datasets).value_counts()
        fig = px.bar(
            x=dataset_counts.index,
            y=dataset_counts.values,
            labels={'x': 'Dataset', 'y': 'Count'},
            title='Distribution of Errors by Dataset'
        )
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)


    def _create_download_section(self):
        """Creates a download button for annotations if they exist in session state"""
        if 'download_ready' not in st.session_state:
            st.session_state.download_ready = False

        if st.session_state.download_ready:
            test_data = {
                "annotations": [
                    {"id": 1, "category": "test category", "dataset": "test dataset"}
                ],
                "total_annotations": 1
            }
            st.markdown("---")

            st.download_button(
                label="Download JSON",
                data=json.dumps(test_data, indent=2),
                file_name=ResultConstants.ANNOTATIONS_FILE,
                mime="text/plain",
            )


class AnnotationManager:
    def __init__(self):
        self.save_dir = ResultConstants.ANNOTATIONS_DIR
        self.annotations = self._load_current_annotations()
        self.filepath = Path(self.save_dir) / ResultConstants.ANNOTATIONS_FILE

    def _load_current_annotations(self):
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading annotations: {e}")
        return {
            "annotations": [],
            "total_annotations": 0
        }

    def save_annotation(self, annotation_data):
        self.annotations["annotations"].append(annotation_data)
        self.annotations["total_annotations"] = len(self.annotations["annotations"])
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.annotations, f, indent=2)
        except Exception as e:
            print(f"Error saving annotation: {e}, filepath: {self.filepath}")

    def clear_annotations(self):
        try:
            self.annotations = {"annotations": [], "total_annotations": 0}
            with open(self.filepath, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to clear annotations: {e}")
            return False
    def get_annotations(self):
        return self.annotations
class ErrorCategories:
    """Constants for error categories"""
    FORMAT_ERROR = "1. Format Error"
    WRONG_REASONING = "2. Wrong Reasoning"
    WRONG_ANNOTATION = "3. Wrong Annotation"
    LACK_KNOWLEDGE = "4. Lack of Knowledge"

    @classmethod
    def get_all_categories(cls):
        return [
            cls.FORMAT_ERROR,
            cls.WRONG_REASONING,
            cls.WRONG_ANNOTATION,
            cls.LACK_KNOWLEDGE
        ]
if __name__ == '__main__':
    dataset_sizes_path = Constants.TemplatesGeneratorConstants.DATASET_SIZES_PATH
    hos = MetaHistogramOfSamples(dataset_sizes_path)
    hos.display_page()
