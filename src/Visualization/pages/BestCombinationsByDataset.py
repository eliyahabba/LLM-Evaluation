import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants
from src.utils.MMLUConstants import MMLUConstants
ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants

file_path = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH / f"{ResultConstants.BEST_COMBINATIONS}.csv"

split_each_mmlu_group_to_train_test = (
    "Split eaxh MMLU group to train and test (for example: mmlu.abstract_algebra and "
    "mmul.college_mathematics)")
split_mmlu_groups_to_train_test = "Split MMLU groups to train and test (for example: math, health, physics)"
split_mmlu_categories_to_train_test = "Split MMLU categories to train and test (for example: STEM, humanities, social sciences, other)"
split_all_datasets_to_train_test = "Split all datasets to train and test by the model"


class BestCombinationsDisplayer:
    def __init__(self):
        self.best_combinations = pd.read_csv(file_path)
        self.best_combinations.drop(columns=['accuracy'], inplace=True)
        self.override_options = ConfigParams.override_options

    def display(self, model_data):
        st.title("Best Combinations")
        # add subtitle that table displays the best combinations for each model and dataset on all the possible axes
        st.write("The table below displays the best combinations for each model and dataset on all the possible axes.")
        # write the table
        mmlu_subcategories = MMLUConstants.SUBCATEGORIES
        mmlu_categories = MMLUConstants.SUBCATEGORIES_TO_CATEGORIES
        model_data['mmlu_subcategories'] = model_data['dataset'].apply(lambda x: (mmlu_subcategories[x.
                                                                          split('mmlu.')[1]][0] if x.startswith(
            'mmlu.') else 'None'))
        model_data['mmlu_categories'] = model_data['dataset'].apply(lambda x: (
            mmlu_categories[mmlu_subcategories[x.
            split('mmlu.')[1]][0]] if x.startswith('mmlu.') else 'None'))
        st.write(model_data)

    def evaluate_by_model_or_dataset(self, model_or_dataset_key: str, model_data, split_option: str=None):
        st.subheader(f"Evaluation by {model_or_dataset_key}")
        # if there are st least 5 rows in the model data, else skip the model
        split_options = [split_each_mmlu_group_to_train_test, split_mmlu_groups_to_train_test,
                         split_mmlu_categories_to_train_test, split_all_datasets_to_train_test]

        split_options = ["subcategories", "categories", "all datasets"]

        if split_option =="subcategories":
            model_datas = {group:
                model_data[model_data['mmlu_subcategories'] == group] for group in model_data['mmlu_subcategories'].unique()}
        elif split_option == "categories":
            model_datas = {group:
                model_data[model_data['mmlu_categories'] == group] for group in
                           model_data['mmlu_categories'].unique()}
        else:
            model_datas = [model_data]
        for group, cur_data in model_datas.items():
            figs = []
            for axis in self.override_options.keys():
                axis_values = self.override_options[axis]
                for i in range(len(axis_values)):
                    if axis_values[i] == '\n':
                        axis_values[i] = '\\n'
                    if axis_values[i] == ' ':
                        axis_values[i] = '\\s'

                # axis_values = [str(value) for value in axis_values]
                # display histogram of the axis values
                fig = plt.figure()
                values = cur_data[axis].values

                # if the axis is bool, convert the values to string
                #count for each value in the axis_values the number of times it appears in the values
                axis_values_counts = {value: np.sum(values == value) for value in axis_values}
                # sort the values by the lexographical order
                axis_values_counts = dict(sorted(axis_values_counts.items()))
                unique_values = list(axis_values_counts.keys())
                value_indices = [unique_values.index(value) for value in values]
                if isinstance(unique_values[0], bool):
                    unique_values = [str(unique_value) for unique_value in unique_values]

                # Create bins such that each value falls into its own bin
                bins = np.arange(len(unique_values) + 1) - 0.5

                # Plot histogram using mapped indices
                plt.hist(value_indices, bins=bins, alpha=0.75, rwidth=0.8)
                # Set ticks at the center of each bin
                bin_centers = np.arange(len(unique_values))
                plt.xticks(bin_centers, labels=unique_values)
                plt.title(f"{axis} histogram")
                # out the tic
                figs.append(fig)
            st.write(f"Group: {group}, Number of samples: {len(values)}")
            cols = st.columns(len(figs))
            # wrtie the
            for i, fig in enumerate(figs):
                with cols[i]:
                    st.pyplot(fig)



    def evaluate(self):
        st.title("Evaluation")
        st.subheader("The table displays the evaluation of the best combinations")
        model_or_dataset = st.radio("Evaluation by model or by dataset?", ("Model", "Dataset"))
        models = self.best_combinations['model'].unique()
        datasets = self.best_combinations['dataset'].unique()
        if model_or_dataset == "Model":
            model = st.selectbox("Choose a model", models)
            model_data = self.best_combinations[self.best_combinations['model'] == model]
            best_combinations_displayer.display(model_data)

            split_options = [split_each_mmlu_group_to_train_test, split_mmlu_groups_to_train_test,
                             split_mmlu_categories_to_train_test, split_all_datasets_to_train_test]
            split_options = ["subcategories", "categories", "all datasets"]
            split_option = st.selectbox("Split the dataset to train and test by:", split_options)
            self.evaluate_by_model_or_dataset("model", model_data, split_option)

        # elif model_or_dataset == "Dataset":
        #     self.evaluate_by_model_or_dataset("dataset")

    def split_to_train_and_test(self,model_or_dataset_key, model_data, random_state, split_option):
        # there is 4 options to split the dat.
        # 1. gropu the mmlu dataset to grous and split the groups to train and test
        # 2. split each group of mmlu dataset to train and test
        # 3. split the mmlu categories to train and test
        # 4. split the dataset to train and test by the model
        # aks the use to choose the option
        st.write(f"Splitting the dataset to train and test by: {split_option}")

        if split_option == split_each_mmlu_group_to_train_test:
            pass
            # split each group of mmlu dataset to train and test
            # train_test_pairs = []
            # for _, data in model_data.groupby('mmlu_group'):
            #     if data.shape[0] == 1:
            #         continue
            #     train_test_pairs.append(train_test_split(data, test_size=0.2, random_state=random_state))
            # self.evaluate_each_group(train_test_pairs)
        elif split_option == split_mmlu_groups_to_train_test:
            # take 0.2 of the groups to test and the rest to train
            groups = model_data['mmlu_subcategories'].unique()
            train_groups, test_groups = train_test_split(groups, test_size=0.2, random_state=random_state)
            train_data_groups = [model_data[model_data['mmlu_subcategories'] == group] for group in train_groups]
            test_data_groups = [model_data[model_data['mmlu_subcategories'] == group] for group in test_groups]
            self.evaluate_groups(train_data_groups, test_data_groups)
        elif split_option == split_mmlu_categories_to_train_test:
            # take 0.2 of the groups to test and the rest to train
            groups = model_data['mmlu_categories'].unique()
            train_groups, test_groups = train_test_split(groups, test_size=0.2, random_state=random_state)
            train_data_groups = [model_data[model_data['mmlu_categories'] == group] for group in train_groups]
            test_data_groups = [model_data[model_data['mmlu_categories'] == group] for group in test_groups]
            self.evaluate_groups(train_data_groups, test_data_groups)
        elif split_option == split_all_datasets_to_train_test:
            train_data, test_data = train_test_split(model_data, test_size=0.2, random_state=random_state)
            self.evaluate_dataset(model_or_dataset_key, train_data, test_data)

    def evaluate_each_group(self, train_test_pairs):
        pass

    def evaluate_groups(self, train_data_groups, test_data_groups):
        pass
    def evaluate_dataset(self, model_or_dataset_key, train_data, test_data):
        best_results = {}
        for axis in self.override_options.keys():
            best_results[axis] = train_data[axis].value_counts().idxmax()

            self.evaluate_model(model_or_dataset_key, best_results, test_data)

    def get_mean_params(self, model_or_dataset_key, model_data,split_option=None):
        # Split the dataset into two parts
        # add bottom the regenrate the train and test data (i.e. generate the random state)
        random_state = 42
        if st.button("Regenerate random seed",
                     key=f"regenerate_{model_data.iloc[0]['model']}_{model_data.iloc[0]['dataset']}"):
            random_state = random.randint(0, 100)
        self.split_to_train_and_test(model_or_dataset_key, model_data, random_state, split_option)

    # Calculate the mean best parameters from the first part of the dataset

    def evaluate_model(self, model_or_dataset_key, best_results, test_data):
        # For each row in the test data, check if the best results match the actual results,
        # and print the predicted and actual results
        correct_predictions = 0
        total_predictions = len(test_data) * len(self.override_options)
        results_df = []
        test_data_axis = test_data[self.override_options.keys()]
        # display the pd.series of the best results
        # display the test data axis when each cell with correct value set with green color and wrong value with red
        # Iterate over each row in the test data
        for index, row in test_data_axis.iterrows():
            # Create a list to store color-coded values for the current row
            row_values, correct_prediction = self.create_row_values(row, best_results)
            results_df.append(row_values)
            correct_predictions += correct_prediction

        # Convert the results list to a DataFrame and display it
        test_data_results = pd.DataFrame(results_df, columns=test_data_axis.columns)
        # merge the index to the dataframe the dataset column in test data
        test_data_results = self.create_test_data_results(test_data, test_data_results, model_or_dataset_key)
        self.display_evaluation_summary(test_data_results, best_results)
        # Display overall evaluation summary
        st.write(f"Correct Predictions: {correct_predictions}/{total_predictions}")

    def create_test_data_results(self, test_data, test_data_results, model_or_dataset_key):
        index = 'dataset' if model_or_dataset_key == 'model' else 'model'
        test_data_results = pd.concat([test_data[index].reset_index(drop=True), test_data_results], axis=1)
        test_data_results.set_index(index, inplace=True)
        test_data_results.index.name = None
        return test_data_results

    def create_row_values(self, row, best_results):
        correct_predictions = 0
        row_values = []
        for axis, actual_result in row.items():
            # Get the predicted result for the current axis
            predicted_result = best_results.get(axis, None)
            # Check if the predicted result matches the actual result
            if predicted_result is not None and predicted_result == actual_result:
                # If the prediction is correct, add the actual result with green color
                row_values.append(f'<span style="color:green">{actual_result}</span>')
                correct_predictions += 1
            else:
                # If the prediction is incorrect, add the actual result with red color
                row_values.append(f'<span style="color:red">{actual_result}</span>')
        # Add the color-coded row values to the results DataFrame
        return row_values, correct_predictions

    def display_evaluation_summary(self, test_data_results, best_results):
        # display the train and test data side by side (use columns)
        # wrtie the current model as the sub header
        col1, col2 = st.columns(2)
        with col1:
            st.write("Train Data:")
            best_df = pd.Series(best_results).to_frame()
            # change the values in the df to green f'<span style="color:green">{actual_result}</span>'
            best_df = best_df.applymap(lambda x: f'<span style="color:green">{x}</span>')
            # remove the index column, and display only the df without the index column
            best_df.index.name = None
            st.markdown(best_df.to_html(escape=False, header=False), unsafe_allow_html=True)
        with col2:
            st.write("Test Data:")
            st.markdown(test_data_results.to_html(escape=False), unsafe_allow_html=True)


if __name__ == "__main__":
    best_combinations_displayer = BestCombinationsDisplayer()
    best_combinations_displayer.evaluate()

