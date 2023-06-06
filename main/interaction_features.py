import pandas as pd
import numpy as np
from itertools import combinations
import os
from dotenv import load_dotenv

load_dotenv()
base_path = os.environ.get(
    "BASE_PATH", "/kaggle/input/icr-identify-age-related-conditions"
)
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv(f"{base_path}/train.csv")
test = pd.read_csv(f"{base_path}/test.csv")

num_cols = train.select_dtypes(include=["float64"]).columns.tolist()
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
cat_cols.remove("Id")


def corr_feat():
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    df = train[num_cols + ["Class"]]
    # Let's assume df is your DataFrame and 'target' is the column with the target variable
    corr_matrix = df.corr().abs()

    top_corr_features = (
        corr_matrix["Class"].sort_values(ascending=False).head(10).index.tolist()
    )

    # Generate all combinations of pairs of these features
    feature_pairs = list(combinations(top_corr_features, 2))

    top_corr_matrix = df[top_corr_features + ["Class"]].corr().abs()
    print(top_corr_matrix)
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr_matrix, annot=True, fmt=".2f")

    # Create interaction features for these pairs
    for pair in feature_pairs:
        df["interaction_{}_{}".format(pair[0], pair[1])] = df[pair[0]] * df[pair[1]]
    plt.show()


# corr_feat()

greeks = pd.read_csv(f"{base_path}/greeks.csv")
alpha = greeks[["Id", "Alpha"]].merge(train, on=["Id"], how="inner")
print(alpha.Alpha.value_counts())
#
gamma = greeks[["Id", "Gamma"]].merge(train, on=["Id"], how="inner")
print(gamma.Gamma.value_counts())


# print(np.full((20, 4), 1/4))
