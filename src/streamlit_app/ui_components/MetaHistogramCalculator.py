import sys
from pathlib import Path

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

import pandas as pd
import streamlit as st

from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants

ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class MetaHistogramCalculator:

    @staticmethod
    @st.cache_data
    def aggregate_data_across_models(selected_results_file, selected_models_files, shot, datasets_names):
        total_merge_df = pd.DataFrame()
        shot_suffix = Path(shot) / "empty_system_format"

        for model_file in selected_models_files:
            mmlu_files = MetaHistogramCalculator.collect_existing_files(selected_results_file, model_file,
                                                                        datasets_names, shot_suffix)
            if mmlu_files:
                merged_df = MetaHistogramCalculator.process_model_files(mmlu_files, model_file)
                total_merge_df = pd.concat([total_merge_df, merged_df], axis=1)

        return total_merge_df

    @staticmethod
    @st.cache_data
    def collect_existing_files(selected_results_file, model_file, datasets_names, shot_suffix):
        mmlu_files = [
            selected_results_file / model_file / Path(datasets_name) / shot_suffix / "comparison_matrix_test_data.csv"
            for datasets_name in datasets_names]
        return [file for file in mmlu_files if file.exists()]

    @staticmethod
    @st.cache_data
    def process_model_files(mmlu_files, model_file):
        merged_df = pd.DataFrame()

        for mmlu_file in mmlu_files:
            df = MetaHistogramCalculator.calculate_prediction_accuracy(mmlu_file)
            df = df.reset_index()
            df['example_number'] = mmlu_file.parents[2].name + "_" + df.index.astype(str)
            merged_df = pd.concat([merged_df, df])

        merged_df.drop(columns=['accuracy', 'num_of_predictions'], inplace=True)
        merged_df.set_index('example_number', inplace=True)
        merged_df.columns = [f"{model_file.name}_{col}" for col in merged_df.columns]

        return merged_df

    @staticmethod
    @st.cache_data
    def calculate_and_add_accuracy_columns(total_merge_df):
        total_merge_df['correct'] = total_merge_df.sum(axis=1, skipna=True)
        total_merge_df['number_of_predictions'] = total_merge_df.count(
            axis=1) - 1  # Exclude the 'correct' column it  
        total_merge_df['accuracy'] = (total_merge_df['correct'] / total_merge_df['number_of_predictions']) * 100
        return total_merge_df

    @staticmethod
    @st.cache_data
    def calculate_prediction_accuracy(results_file: Path):
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
        df.index.name = 'example_number'
        return df

    @staticmethod
    @st.cache_data
    def extract_example_data(examples):
        examples_id = examples.index.to_series().str.rsplit('_', n=1, expand=True)
        example_data = pd.DataFrame({
            'dataset': examples_id[0],
            'example_number': examples_id[1],
            'accuracy': examples['accuracy']
        })
        return example_data
