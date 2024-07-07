import pandas as pd
import pingouin as pg
from scipy.stats import kendalltau

# Taking values from the above example in Lists
X = [1, 2, 3, 4, 5, 6, 7]
Y = [1, 3, 6, 2, 7, 4, 5]

# Calculating Kendall Rank correlation
corr, _ = kendalltau(X, Y)

# Example data
data = {
    'Item 1': [1, 2, 3, 4],
    'Item 2': [2, 3, 1, 4],
    'Item 3': [3, 1, 2, 4],
    'Item 4': [4, 2, 3, 1]
}
df = pd.DataFrame(data)

# Calculate Kendall's W
kendall_w_result = pg.kendall_w(df)

print(kendall_w_result)
