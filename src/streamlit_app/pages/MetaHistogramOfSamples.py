from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants

MMLUConstants = Constants.MMLUConstants
ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class MetaHistogramOfSamples:
    def get_files(self):
        selected_results_file = ResultsLoader.get_folder_selections_options(MAIN_RESULTS_PATH,
                                                                            "Select results folder to visualize",
                                                                            reverse=True)
        shot = st.sidebar.selectbox("select number of shots", ["zero_shot", "three_shot"])

        # select all the possible models:
        folders_names = [file for file in selected_results_file.iterdir() if file.is_dir()]
        names_to_display = {f.name: f for f in folders_names}
        names_to_display = dict(sorted(names_to_display.items(), key=lambda x: (x[0].lower(), x[0]), reverse=False))
        # multi select the models
        text_to_display = "Select models to visualize"
        models = st.sidebar.multiselect(text_to_display, list(names_to_display.keys()), default=list(names_to_display.keys()))
        # models = st.sidebar.multiselect(text_to_display, list(names_to_display.keys()), default=[])
        selected_models_files = [names_to_display[model] for model in models]

        return selected_results_file, selected_models_files, shot

    def display_page(self):
        st.title("Histogram of Samples")
        selected_results_file, selected_models_files, shot = self.get_files()



        self.new_display_aggregated_results(selected_results_file, selected_models_files, shot)

    def new_display_aggregated_results(self, selected_results_file, selected_models_files, shot):
        split_option = st.selectbox("aggregated the dataset by:", MMLUConstants.SPLIT_OPTIONS)
        data_options = MMLUSplitter.get_data_options(split_option)
        split_option_value = st.selectbox("select the split option value:", data_options)
        datasets_names = MMLUSplitter.get_data_files(split_option, split_option_value)
        shot_suffix = Path(shot) / Path("empty_system_format")
        total_merge_df = pd.DataFrame()
        for model_file in selected_models_files:
            mmlu_files = [selected_results_file/ model_file / Path(datasets_name) / shot_suffix
                          / Path("comparison_matrix_test_data.csv") for datasets_name in
                          datasets_names]
            # filter non existing files
            mmlu_files = [file for file in mmlu_files if file.exists()]
            if len(mmlu_files) > 0:
                merged_df = pd.DataFrame()
                for mmlu_file in mmlu_files:
                    df = self.display_samples_prediction_accuracy(mmlu_file, display_results=False)
                    # reset the index and add prfix mmlu_file.parents[2].name + the index of the row
                    df = df.reset_index()
                    df['example number'] = mmlu_file.parents[2].name + "_" + df.index.astype(str)
                    merged_df = pd.concat([merged_df, df])
            # concat on the example number column such that sum the values of the same example number to each cell
                # reomove the col accuarcy and num_of_predictions
                merged_df = merged_df.drop(columns=['accuracy', 'num_of_predictions'])
                # set the index to example number
                merged_df = merged_df.set_index('example number')
                # add to each column the name of the model
                merged_df.columns = [model_file.name + "_" + col for col in merged_df.columns]
                total_merge_df = pd.concat([total_merge_df, merged_df], axis=1)
        # total_merge_df = total_merge_df.groupby('example number').sum().reset_index()
        # ad accuracy column and the values is the sum of this row
        total_merge_df['correct'] = total_merge_df.sum(axis=1, skipna=True)
        total_merge_df['number_of_predictions'] = total_merge_df.count(axis=1) - 1  # Subtract 1 to exclude the 'accuracy' column itself
        total_merge_df['accuracy'] = (total_merge_df['correct'] / total_merge_df['number_of_predictions']) * 100
        # add the accuracy column
        self.plot_aggregated_histogram(total_merge_df, split_option, split_option_value)


    def display_samples_prediction_accuracy(self, results_file: Path, display_results=True):
        """
        Display the results of the model.

        @param results_file: the path to the results file
        @return: None
        """
        df = pd.read_csv(results_file)
        # sum each row to get the total number of instances (sum the ones in the row and divide by the number of
        # ones + zeros)
        predictions_columns = [col for col in df.columns if "experiment_template" in col]
        df['count_true_preds'] = df[predictions_columns].sum(axis=1)
        df['num_of_predictions'] = df[predictions_columns].notnull().sum(axis=1)
        # count the values for each row
        df['accuracy'] = df['count_true_preds'] / df['num_of_predictions']
        # multiply the accuracy by 100
        df['accuracy'] = round(df['accuracy'] * 100, 2)
        # put the accuracy in the first column
        df = df[['num_of_predictions', 'accuracy'] + predictions_columns]
        # add name to the index column
        df.index.name = 'example number'
        if display_results:
            st.write(df)
        return df

    def plot_histogram(self, df, title="Histogram of Prediction Accuracy"):
        """
        Plot the histogram of the results.
        @param df: DataFrame containing the data to plot.
        """
        st.markdown(f"There are {len(df)} examples in the dataset.")
        # Setting up the plot
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.grid(True)  # Add grid lines for better readability
        ax.set_axisbelow(True)  # Ensure grid lines are behind other plot elements

        # Plotting the histogram
        bins = np.arange(0, 105, 5)  # Adjust bins to include the range from 12 to 100
        df['accuracy'].plot(kind='hist', bins=bins, ax=ax, color='skyblue', edgecolor='black')

        # Adding title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Percentage of Templates with Correct Predictions", fontsize=14)
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
        # plt.show()
        st.pyplot(fig)

    def plot_aggregated_histogram(self, merged_df, split_option, split_option_value):
        value = f"on {split_option_value}" if split_option_value else ""
        self.plot_histogram(merged_df, title=f"Aggregated Histogram by {split_option} {value}")


if __name__ == '__main__':
    hos = MetaHistogramOfSamples()
    hos.display_page()
