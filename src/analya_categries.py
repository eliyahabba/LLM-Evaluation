import os

import pandas as pd

if __name__ == '__main__':
    file_name = "data1_output"
    full_file_name = f"{file_name}.csv"
    df1 = pd.read_csv(full_file_name, index_col=0)

    file_name2 = "data2_output"
    full_file_name = f"{file_name2}.csv"
    df2 = pd.read_csv(full_file_name, index_col=0)

    df = pd.concat([df1, df2])

    suggest_categories = ["Animal Behavior", "Biological Principles",
                          "Physical Principles", "Cultural Principles",
                          "Social Conventions",
                          "Engineering and Architectural Features",
                          "Nutritional and Food Practices",
                          "Religion Principles",
                          "Safety and Survival Situations", "Temporal Principles",
                          "Weather and Environmental Conditions",
                          "Geographic Knowledge", "Historical Contexts",
                          "Traffic Regulations and Conventions",
                          "Object Counting and Placement", "Law, Order and Crime", "Education and Study Environment",
                          "Office and Equipment Troubleshooting", "Game Strategy and Tactics",
                          "Health and Personal Care",
                          "Sport"]

    # count how many each category appears in category column (There can be
    # catgories that are not in the df, so the count will be 0)
    category_count = {}
    for category in suggest_categories:
        category_count[category] = df[df["category"] == category]['category'].count()

    # create a new dataframe with the category count
    df_category_count = pd.DataFrame(category_count, index=[0]).T
    # rename the column to "number of occurrences"
    df_category_count.columns = ["number of occurrences"]

    # sort the df by the count
    df_category_count.sort_values(by="number of occurrences", axis=0, ascending=False, inplace=True)

    # save the dataframe to a csv file

    df_category_count.to_csv("category_count.csv")
    print("Category count saved to category_count.csv")

    # print the category count sorted by the count
    print(df_category_count)
    #
    # plot a bar chart of the category count
    import matplotlib.pyplot as plt

    df_category_count.plot(kind='bar', figsize=(10, 10))
    plt.xlabel("Category")
    plt.ylabel("Frequency")
    plt.title("Category Frequency")
    # wrap the x labels so they are readable wrap=True
    from textwrap import wrap
    # remove the legend
    plt.legend().remove()
    # set the x-axis labels to take more Area from the figure, so the labels are readable
    plt.subplots_adjust(bottom=0.35)  # Adjust bottom margin
    # add the numbers of each bar as text on the top of the bar
    for i in range(len(df_category_count)):
        plt.text(i, df_category_count.iloc[i, 0], df_category_count.iloc[i, 0], ha='center', va='bottom')
    plt.show()

