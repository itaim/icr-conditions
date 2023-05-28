import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

tqdm.pandas()
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint


train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
greeks = pd.read_csv("../data/greeks.csv")
sample_submission = pd.read_csv("../data/sample_submission.csv")

num_cols = test.select_dtypes(include=["float64"]).columns.tolist()
cat_cols = test.select_dtypes(include=["object"]).columns.tolist()
cat_cols.remove("Id")


def preprocess_df(df, encoder=None):
    # Combine numeric and categorical features
    FEATURES = num_cols + cat_cols

    # Fill missing values with mean for numeric variables
    imputer = SimpleImputer(strategy="mean")
    numeric_df = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols)

    # Scale numeric variables using min-max scaling
    scaler = MinMaxScaler()
    scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=num_cols)

    # Encode categorical variables using one-hot encoding
    if not encoder:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded_cat_df = pd.DataFrame(
            encoder.fit_transform(df[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
        )
    else:
        encoded_cat_df = pd.DataFrame(
            encoder.transform(df[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
        )

    # Concatenate the scaled numeric and encoded categorical variables
    return pd.concat([scaled_numeric_df, encoded_cat_df], axis=1), encoder


def xgb_cv():
    FOLDS = 10
    SEED = 1004
    xgb_models = []
    xgb_oof = []
    f_imp = []

    counter = 1
    X = preprocess_df(train)
    y = train["Class"]
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if (fold + 1) % 5 == 0 or (fold + 1) == 1:
            print(f'{"#" * 24} Training FOLD {fold + 1} {"#" * 24}')

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
        watchlist = [(X_train, y_train), (X_valid, y_valid)]

        # XGboost model and fit
        model = XGBClassifier(
            n_estimators=1000, n_jobs=-1, max_depth=4, eta=0.2, colsample_bytree=0.67
        )
        model.fit(
            X_train, y_train, eval_set=watchlist, early_stopping_rounds=300, verbose=0
        )

        val_preds = model.predict_proba(X_valid)[:, 1]
        val_score = log_loss(y_valid, val_preds)
        val_f1 = f1_score(y_valid, [round(p) for p in val_preds])
        best_iter = model.best_iteration

        idx_pred_target = np.vstack([val_idx, val_preds, y_valid]).T
        f_imp.append({i: j for i, j in zip(X.columns, model.feature_importances_)})
        print(
            f'{" " * 20} Log-loss: {val_score:.5f} {" " * 6} best iteration: {best_iter} f1 score: {val_f1}'
        )

        xgb_oof.append(idx_pred_target)
        xgb_models.append(model)

    print("*" * 45)
    print(
        f"Mean Log-loss: {np.mean([log_loss(item[:, 2], item[:, 1]) for item in xgb_oof]):.5f}"
    )


def run_main():
    parameters = {
        "max_depth": randint(3, 6),
        "eta": uniform(0.1, 0.3),
        "colsample_bytree": uniform(0.5, 0.8),
    }

    xgb = XGBClassifier(n_estimators=1000, n_jobs=-1)
    clf = RandomizedSearchCV(
        xgb, parameters, n_iter=20, cv=5, scoring="neg_log_loss", random_state=42
    )

    # Train-test split for model evaluation
    X, encoder = preprocess_df(train)
    y = train["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit the model to the training data and find the best hyperparameters
    clf.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Train a model on the full training set with the best hyperparameters
    best_params = clf.best_params_
    xgb = XGBClassifier(n_estimators=1000, n_jobs=-1, **best_params)
    xgb.fit(X, y)

    # Make predictions on the test set
    test_processed, _ = preprocess_df(test, encoder)
    preds = xgb.predict_proba(test_processed)

    # Create a submission dataframe and save it to a csv file
    submission = pd.DataFrame(preds, columns=["class_0", "class_1"])
    submission.insert(0, "Id", test["Id"])
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    print(train.Class.value_counts())
    print(len(train.columns))
    print(len(test.columns))
    # run_main()
