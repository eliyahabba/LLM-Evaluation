from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants


class DataMissing:
    def __init__(self, summarize_df_path: Path = ResultConstants.SUMMARIZE_DF_PATH):
        self.summarize_df_path = summarize_df_path

    def read_data(self):
        self.df = pd.read_csv(self.summarize_df_path, index_col=[0, 1, 2, 3])
        # the df is a multi index df with thie fromat:
        # index: Model               Dataset               Shots      Configuration
        # columns: Total Data,Data Acquired
        # sum the number of Data Acquired (this is a int column)
        sum_of_data_acquired = self.df["Data Acquired"].sum()
        # this is a bit number so we need to format it with commas
        sum_of_data_acquired = "{:,}".format(sum_of_data_acquired)
        st.markdown(f"Total number of data acquired: {sum_of_data_acquired}")

    def count_missing_data_by_model_and_shots(self):
        # count the number of missing data by model and shots (Data Acquired % Total Data)
        # Data_Acquired = self.df.groupby(["Model", "Shots"])["Data Acquired"].sum()
        # Total_Data = self.df.groupby(["Model", "Shots"])["Total Data"].sum()
        # group_df = self.df.groupby(["Model", "Shots"]).size().reset_index(name='counts')
        # group_df["Data Acquired % Total Data"] = Data_Acquired / Total_Data
        # group_df["Data Acquired % Total Data"] = group_df["Data Acquired % Total Data"].apply(lambda x: round(x, 2))

        Data_Acquired = self.df.groupby(["Model", "Shots"])["Data Acquired"].sum()
        Total_Data = self.df.groupby(["Model", "Shots"])["Total Data"].sum()

        group_df = pd.DataFrame({
            "Data Acquired": Data_Acquired,
            "Total Data": Total_Data
        }).reset_index()

        group_df["Data Acquired % Total Data"] = (group_df["Data Acquired"] / group_df["Total Data"]).apply(lambda x: round(x, 2)) * 100
        # sort the df by the Data Acquired % Total Data
        group_df = group_df.sort_values(by="Data Acquired % Total Data", ascending=False, ignore_index=True)
        # reset the index
        st.write(group_df)

    def filter_by_model(self):
        # the user need to select the model from the list of models
        models = self.df.index.get_level_values("Model").unique().values
        model = st.selectbox("Select the model", models)

        shots = self.df.index.get_level_values("Shots").unique().values
        shots = st.selectbox("Select the shots", shots)

        # filter the df by the model and the shots
        curr_df = self.df.xs((model, shots), level=('Model', 'Shots'))
        st.markdown(f"Model: {model}, Shots: {shots}")
        st.write(curr_df)
        # take only the uncompleted data
        uncompleted_data = curr_df[curr_df["Data Acquired"] != curr_df["Total Data"]]
        st.markdown("Uncompleted data")
        st.write(uncompleted_data)


if __name__ == "__main__":
    data = DataMissing(ResultConstants.SUMMARIZE_DF_PATH)
    data.read_data()
    data.count_missing_data_by_model_and_shots()
    data.filter_by_model()
