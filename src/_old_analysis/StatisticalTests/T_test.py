from itertools import combinations


import scipy.stats as stats

import pandas as pd
from scipy.stats import ttest_rel

df = pd.read_csv('data.csv')
# Assuming df is your DataFrame containing the results
options = df.index[:-1]  # Exclude 'Average' row
alpha = 0.05

# Perform pairwise t-tests and store p-values
p_values = {}
for option1, option2 in combinations(options, 2):
    data1 = df.loc[option1]
    data2 = df.loc[option2]
    t_statistic, p_value = ttest_rel(data1, data2)

# Apply Bonferroni correction
bonferroni_alpha = alpha / len(p_values)

# Identify significant differences
significant_differences = [(option1, option2) for (option1, option2), p_value in p_values.items() if p_value < bonferroni_alpha]

# Output significant differences
print("Significant differences (Bonferroni corrected):")
for option1, option2 in significant_differences:
    print(f"{option1} is significantly different from {option2}")

# Rank the options based on mean performance
mean_performance = df.iloc[:-1].mean(axis=1)
ranked_options = mean_performance.sort_values(ascending=False)

# Output ranked options
print("\nRanked options:")
for i, option in enumerate(ranked_options.index):
    print(f"{i+1}. {option}")

# Identify the best option
best_option = ranked_options.index[0]
print(f"\nBest performing option: {best_option}")

# Assuming your data is stored in a DataFrame called 'df'
heatmap_df = df

# Get list of parameter names (excluding 'Average' row)
parameters = heatmap_df.index.tolist()

# Perform paired t-tests for each pair of parameters
for i in range(len(parameters)):
    for j in range(i + 1, len(parameters)):
        param1 = parameters[i]
        param2 = parameters[j]
        t_statistic, p_value = stats.ttest_rel(heatmap_df.loc[param1], heatmap_df.loc[param2])
        print(f"Paired T-test between {param1} and {param2}:")
        print("T-statistic:", t_statistic)
        print("P-value:", p_value)
        if p_value < 0.05:
            print("There is a significant difference between the parameters.")
        else:
            print("There is no significant difference between the parameters.")
        print()
