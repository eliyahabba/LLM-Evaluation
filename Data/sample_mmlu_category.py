import pandas as pd
import random

# Step 1: Load the CSV file with subject information
csv_file = "mmlu_metadata.csv"  # Replace with your actual CSV file path
subjects_df = pd.read_csv(csv_file)

# Step 2: Group by 'Sub_Category' and select one subject per subcategory
selected_subjects = subjects_df.groupby("Sub_Category").apply(
    lambda group: group.sample(n=1, random_state=42)  # Randomly sample one subject per subcategory
).reset_index(drop=True)

# Step 3: Display the selected subjects
print("Selected subjects, one per subcategory:")
print(selected_subjects)

# Step 4: Save the selected subjects to a new CSV file for further use (optional)
selected_subjects.to_csv("selected_subjects.csv", index=False)

# Step 5: Print each selected subject
for _, row in selected_subjects.iterrows():
    print(f"Sub-Category: {row['Sub_Category']}")
    print(f"Selected Subject: {row['Name']}")
    print(f"Category: {row['Category']}")
    print("------")