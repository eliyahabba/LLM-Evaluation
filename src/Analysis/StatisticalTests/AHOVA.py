import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Create a list to store all the datasets
all_datasets = []

def generate_heatmap(result_file, metadata_df: pd.DataFrame, axis_options: list, selected_values: dict) -> pd.DataFrame:
    # choose the relevant rows from the metadata_df
    for option, value in selected_values.items():
        metadata_df = metadata_df[metadata_df[option] == value]

    # now we have the relevant rows, we need to choose the relevant columns for the heatmap
    # create 2d matrix when the rows are the values of the first axis and the columns are the values of the second axis
    # and the values are the accuracy of the template
    df = pd.read_csv(result_file)

    # add the 'accuracy' columns from df to the metadata_df by the template_name
    metadata_df = metadata_df.join(df.set_index('template_name')['accuracy'], on='template_name')
    heatmap_df = metadata_df.pivot_table(index=axis_options[0], columns=axis_options[1], values='accuracy')
    # add the average of the accuracy for each row and column
    # heatmap_df['Average'] = heatmap_df.mean(axis=1)
    # heatmap_df.loc['Average'] = heatmap_df.mean(axis=0)
    # # the last row and column cell should be empty
    # heatmap_df.loc['Average', 'Average'] = None
    return heatmap_df

def creat_heatmap(metadata_file, result_file):
    # metadata_file = templates_path / self.dataset_file_name / "templates_metadata.csv"
    metadata_df = pd.read_csv(metadata_file, index_col='template_name')
    axis_options = ["choices_seperator", "enumerator"]
    selected_values = {"shuffle_choices": False}
    heatmap_df = generate_heatmap(result_file, metadata_df, axis_options, selected_values)
    return heatmap_df

results_file = r'/cs/labs/gabis/eliyahabba/LLM-Evaluation/results/structured_input/gemma-7b-it/sciq/zero_shot/empty_system_format/cards.sciq_scores_test_data.csv'
metadata_file = r'/cs/labs/gabis/eliyahabba/LLM-Evaluation/Data/MultipleChoiceTemplates/sciq/templates_metadata.csv'
results_file2 = r'/cs/labs/gabis/eliyahabba/LLM-Evaluation/results/structured_input/gemma-7b-it/race_all/zero_shot/empty_system_format/cards.race_all_scores_test_data.csv'
metadata_file2 = r'/cs/labs/gabis/eliyahabba/LLM-Evaluation/Data/MultipleChoiceTemplates/race_all/templates_metadata.csv'

results_files = [results_file, results_file2]
metadata_files = [metadata_file, metadata_file2]
heatmap_dfs = [creat_heatmap(metadata_file, results_file) for metadata_file, results_file in zip(metadata_files, results_files)]
a=1



for df in heatmap_dfs:
    # Melt the DataFrame to create a long format
    melted_df = pd.melt(df.reset_index(), id_vars='choices_seperator', var_name='enumerator', value_name='accuracy')
    # Add a 'dataset' column to identify the dataset
    melted_df['dataset'] = df.columns[-1]

    # Append the melted DataFrame to the list
    all_datasets.append(melted_df)

# Concatenate all the melted DataFrames into a single DataFrame
data = pd.concat(all_datasets, ignore_index=True)

# Fit the two-way ANOVA model
model = ols('accuracy ~ C(choices_seperator) + C(enumerator) + C(choices_seperator):C(enumerator)', data=data).fit()
sm.stats.anova_lm(model, typ=2)

# Print the ANOVA summary
print(model.summary())