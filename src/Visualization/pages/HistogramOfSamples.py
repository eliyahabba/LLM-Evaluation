from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.Visualization.ResultsLoader import ResultsLoader


class HistogramOfSamples:
    def display_page(self):
        st.title("Histogram of Samples")
        dataset_file_name, selected_shot_file_name = ResultsLoader.select_experiment_params()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, result_file = ResultsLoader.select_result_file(result_files, "accuracy")

        df = self.display_samples_prediction_accuracy(result_file)
        self.plot_histogram(df)

        ResultsLoader.display_sample_examples(selected_shot_file_name, dataset_file_name, result_file_name)

    def display_samples_prediction_accuracy(self, results_file: Path):
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
        st.write(df)
        return df

    def plot_histogram(self, df):
        """
        Plot the histogram of the results.
        @param df: DataFrame containing the data to plot.
        """
        # Setting up the plot
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.grid(True)  # Add grid lines for better readability
        ax.set_axisbelow(True)  # Ensure grid lines are behind other plot elements

        # Plotting the histogram
        bins = np.arange(0, 105, 5)  # Adjust bins to include the range from 12 to 100
        df['accuracy'].plot(kind='hist', bins=bins, ax=ax, color='skyblue', edgecolor='black')

        # Adding title and labels
        ax.set_title("Histogram of Prediction Accuracy", fontsize=16)
        ax.set_xlabel("Number of Templates with Correct Prediction", fontsize=14)
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


if __name__ == '__main__':
    hos = HistogramOfSamples()
    hos.display_page()
