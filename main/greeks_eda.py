import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
base_path = os.environ.get(
    "BASE_PATH", "/kaggle/input/icr-identify-age-related-conditions"
)
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv(f"{base_path}/train.csv")
test = pd.read_csv(f"{base_path}/test.csv")
greeks = pd.read_csv(f"{base_path}/greeks.csv")
greeks = greeks.merge(train[["Id", "Class"]], on=["Id"], how="inner").drop(
    columns=["Id"]
)
print(train.Class.value_counts())
num_cols = train.select_dtypes(include=["float64"]).columns.tolist()
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
cat_cols.remove("Id")

greek_feat = greeks
# select_dtypes(include=['object']).columns.tolist()
print(greek_feat.Class.value_counts())
for col in greek_feat.columns:
    print(f"col {col}")
    print(greek_feat[col].value_counts())


# print(greek_feat.describe())
def preprocess_greek() -> pd.DataFrame:
    pass


def eda():
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle("Numerical Attributes vs. Target")
    sns.set_style("whitegrid")

    sns.stripplot(
        x=greek_feat.Delta, y=greek_feat.Class, linewidth=0.6, jitter=0.3, ax=axes[0, 0]
    )
    print(greek_feat.Delta.value_counts())
    # print(greek_feat[(greek_feat.Alpha == 'C')].Class.value_counts())
    # print(len(greek_feat[(greek_feat.Alpha != 'A') & (greek_feat.Class == 0)].index))
    # print(len(greek_feat[(greek_feat.Alpha != 'A') & (greek_feat.Class == 0)].index))

    # print(greek_feat[(greek_feat.Delta == 'B')].Class.value_counts())
    # print(greek_feat[(greek_feat.Delta == 'A')].Class.value_counts())
    # print(greek_feat[(greek_feat.Delta == 'C')].Class.value_counts())
    # print(greek_feat[(greek_feat.Delta == 'D')].Class.value_counts())
    # print(greek_feat[(greek_feat.Beta=='C')].Class.value_counts())
    # print(greek_feat[(greek_feat.Beta=='B')].Class.value_counts())
    # print(greek_feat[(greek_feat.Beta=='A')].Class.value_counts())
    #
    # print(len(greek_feat[(greek_feat.Gamma.isin(['M','N']))&(greek_feat.Class==1)].index))
    # print(len(greek_feat[(~greek_feat.Gamma.isin(['M','N']))&(greek_feat.Class==0)].index))
    # print(len(greek_feat[(greek_feat.Alpha != 'A') & (greek_feat.Class == 0)].index))
    # sns.stripplot(
    #     x=greek_feat.Delta, y=greek_feat.Class, linewidth=0.6, jitter=0.3, ax=axes[0, 1]
    # )

    # sns.stripplot(x=greek_feat.Gamma, y=greek_feat.Class, linewidth=0.6, jitter=0.3, ax=axes[0, 1])
    #
    # sns.stripplot(x=greek_feat.Delta, y=greek_feat.Class, linewidth=0.6, jitter=0.3, ax=axes[0, 1])

    plt.show()


# Alpha - binary
# Gamma - binary
eda()

g_cat_cols = ["Alpha", "Beta", "Gamma", "Delta"]
# p
greeks = greeks[g_cat_cols + ["Class"]]


def corr_heat():
    greeks[g_cat_cols] = greeks[g_cat_cols].astype("category")
    greeks[g_cat_cols] = greeks[g_cat_cols].apply(lambda x: x.cat.codes)
    greeks_corr = greeks.corr()
    print(greeks_corr)

    sns.heatmap(
        greeks_corr, xticklabels=greeks_corr.columns, yticklabels=greeks_corr.columns
    )
    plt.show()


# corr_heat()
print(len(num_cols))
