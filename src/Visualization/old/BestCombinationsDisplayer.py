# import sys
# from pathlib import Path
#
# import pandas as pd
# import streamlit as st
# from sklearn.model_selection import train_test_split
# file_path = Path(__file__).parents[3]
# sys.path.append(str(file_path))
#
# from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
# from src.utils.Constants import Constants
#
# ExperimentConstants = Constants.ExperimentConstants
# ResultConstants = Constants.ResultConstants
#
# file_path = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH / f"{ResultConstants.BEST_COMBINATIONS}.csv"
#
#
#
# class BestCombinationsDisplayer:
#     def __init__(self):
#         self.best_combinations = pd.read_csv(file_path)
#         self.override_options = ConfigParams.override_options
#
#     def display(self):
#         st.title("Best Combinations")
#         # add subtitle that table displays the best combinations for each model and dataset on all the possible axes
#         st.write("The table below displays the best combinations for each model and dataset on all the possible axes.")
#         # write the table
#         st.write(self.best_combinations)
#
#     def evaluate_by_model_or_dataset(self, model_or_dataset_key: str):
#         st.subheader(f"Evaluation by {model_or_dataset_key}")
#         model_results = {}
#         for model_or_dataset in self.best_combinations[model_or_dataset_key].unique():
#             st.markdown(f"#### {model_or_dataset_key}: {model_or_dataset}")
#             current_data = self.best_combinations[self.best_combinations[model_or_dataset_key] == model_or_dataset]
#             # if there are st least 5 rows in the model data, else skip the model
#             if len(current_data) < 5:
#                 continue
#             best_results, test_data = self.get_mean_params(current_data)
#             # Get mean best params for this model
#             # Evaluate the model with mean params on the test data
#             evaluation_result = self.evaluate_model(model_or_dataset_key, best_results, test_data)
#             model_results[model_or_dataset] = evaluation_result
#
#     def evaluate(self):
#         st.title("Evaluation")
#         st.subheader("The table displays the evaluation of the best combinations")
#         model_or_dataset = st.radio("Evaluation by model or by dataset?", ("Model", "Dataset"))
#
#         if model_or_dataset == "Model":
#             self.evaluate_by_model_or_dataset("model")
#
#         elif model_or_dataset == "Dataset":
#             self.evaluate_by_model_or_dataset("dataset")
#
#     def get_mean_params(self, model_data):
#         # Split the dataset into two parts
#         # add bottom the regenrate the train and test data (i.e. generate the random state)
#         random_state = 42
#         if st.button("Regenerate random seed",
#                      key=f"regenerate_{model_data.iloc[0]['model']}_{model_data.iloc[0]['dataset']}"):
#             import random
#             random_state = random.randint(0, 100)
#         train_data, test_data = train_test_split(model_data, test_size=0.2, random_state=random_state)
#
#         # Calculate the mean best parameters from the first part of the dataset
#         best_results = {}
#         for axis in self.override_options.keys():
#             best_results[axis] = train_data[axis].value_counts().idxmax()
#         return best_results, test_data
#
#     def evaluate_model(self, model_or_dataset_key, best_results, test_data):
#         # For each row in the test data, check if the best results match the actual results,
#         # and print the predicted and actual results
#         correct_predictions = 0
#         total_predictions = len(test_data) * len(self.override_options)
#         results_df = []
#         test_data_axis = test_data[self.override_options.keys()]
#         # display the pd.series of the best results
#         # display the test data axis when each cell with correct value set with green color and wrong value with red
#         # Iterate over each row in the test data
#         for index, row in test_data_axis.iterrows():
#             # Create a list to store color-coded values for the current row
#             row_values, correct_prediction = self.create_row_values(row, best_results)
#             results_df.append(row_values)
#             correct_predictions += correct_prediction
#
#         # Convert the results list to a DataFrame and display it
#         test_data_results = pd.DataFrame(results_df, columns=test_data_axis.columns)
#         # merge the index to the dataframe the dataset column in test data
#         test_data_results = self.create_test_data_results(test_data, test_data_results, model_or_dataset_key)
#         self.display_evaluation_summary(test_data_results, best_results)
#         # Display overall evaluation summary
#         st.write(f"Correct Predictions: {correct_predictions}/{total_predictions}")
#
#     def create_test_data_results(self, test_data, test_data_results, model_or_dataset_key):
#         index = 'dataset' if model_or_dataset_key == 'model' else 'model'
#         test_data_results = pd.concat([test_data[index].reset_index(drop=True), test_data_results], axis=1)
#         test_data_results.set_index(index, inplace=True)
#         test_data_results.index.name = None
#         return test_data_results
#
#     def create_row_values(self, row, best_results):
#         correct_predictions = 0
#         row_values = []
#         for axis, actual_result in row.items():
#             # Get the predicted result for the current axis
#             predicted_result = best_results.get(axis, None)
#             # Check if the predicted result matches the actual result
#             if predicted_result is not None and predicted_result == actual_result:
#                 # If the prediction is correct, add the actual result with green color
#                 row_values.append(f'<span style="color:green">{actual_result}</span>')
#                 correct_predictions += 1
#             else:
#                 # If the prediction is incorrect, add the actual result with red color
#                 row_values.append(f'<span style="color:red">{actual_result}</span>')
#         # Add the color-coded row values to the results DataFrame
#         return row_values, correct_predictions
#
#     def display_evaluation_summary(self, test_data_results, best_results):
#         # display the train and test data side by side (use columns)
#         # wrtie the current model as the sub header
#         col1, col2 = st.columns(2)
#         with col1:
#             st.write("Train Data:")
#             best_df = pd.Series(best_results).to_frame()
#             # change the values in the df to green f'<span style="color:green">{actual_result}</span>'
#             best_df = best_df.applymap(lambda x: f'<span style="color:green">{x}</span>')
#             # remove the index column, and display only the df without the index column
#             best_df.index.name = None
#             st.markdown(best_df.to_html(escape=False, header=False), unsafe_allow_html=True)
#         with col2:
#             st.write("Test Data:")
#             st.markdown(test_data_results.to_html(escape=False), unsafe_allow_html=True)
#
#
# if __name__ == "__main__":
#     best_combinations_displayer = BestCombinationsDisplayer()
#     best_combinations_displayer.display()
#     best_combinations_displayer.evaluate()
