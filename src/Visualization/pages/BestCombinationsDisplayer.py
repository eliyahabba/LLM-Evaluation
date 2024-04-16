from collections import Counter

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants

file_path = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH / f"{ResultConstants.BEST_COMBINATIONS}.csv"


class BestCombinationsDisplayer:
    def __init__(self):
        self.best_combinations = pd.read_csv(file_path)

    def display(self):
        st.title("Best Combinations")
        # add subtitle that table displays the best combinations for each model and dataset on all the possible axes
        st.write("The table below displays the best combinations for each model and dataset on all the possible axes.")
        # write the table
        st.write(self.best_combinations)

    def evaluate(self):
        st.title("Evaluation")
        st.subheader("The table displays the evaluation of the best combinations")
        model_or_dataset = st.radio("Evaluation by model or by dataset?", ("Model", "Dataset"))

        if model_or_dataset == "Model":
            st.subheader("Evaluation by model")
            model_results = {}
            for model in self.best_combinations['Model'].unique():
                model_data = self.best_combinations[self.best_combinations['Model'] == model]
                mean_params = self.get_mean_params(model_data)  # Get mean best params for this model
                # Evaluate the model with mean params on the test data
                evaluation_result = self.evaluate_model(model, mean_params)
                model_results[model] = evaluation_result
            st.write(model_results)

        elif model_or_dataset == "Dataset":
            st.subheader("Evaluation by dataset")
            # Perform evaluation by dataset

    def get_mean_params(self, model_data):
        # Split the dataset into two parts
        train_data, test_data = train_test_split(model_data, test_size=0.5, random_state=42)
        # Calculate the mean best parameters from the first part of the dataset
        mean_params = train_data['Best_Params'].mean()
        return mean_params, test_data

    def evaluate_model(self, model, params_test_data):
        mean_params, test_data = params_test_data
        # Placeholder for evaluating the model with mean params on the test data
        # Implement your evaluation logic here, using test_data
        return f"Evaluation result for {model} with mean params {mean_params} on the test data"

if __name__ == "__main__":
    BestCombinationsDisplayer().display()
