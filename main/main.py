import os
import pickle
import pprint
import time
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from auxiliary_models import (
    add_feature_predictions,
    train_meta_model,
    merge_features_df,
    add_feature_values,
)
from hyperparameters import HyperparametersOptimizer
from metrics import get_scale_pos
from model import Bundle, VERSION, ICRData, ModelResult, get_num_cols, get_cat_cols
from model_trainer import GradientBoosterTrainer, BinaryClassifier

# from hyperparameters import HyperparametersOptimizer
# from model import VERSION, Bundle
# from model_trainer import GradientBoosterTrainer

load_dotenv()

base_path = os.environ.get(
    "BASE_PATH", "/kaggle/input/icr-identify-age-related-conditions"
)


def load_data() -> ICRData:
    return ICRData(
        pd.read_csv(f"{base_path}/train.csv"), pd.read_csv(f"{base_path}/greeks.csv")
    )


@dataclass
class CompositeModel:
    auxiliary: Dict[str, ModelResult]
    primary: ModelResult


def save_models_results(models: CompositeModel):
    results_path = os.environ.get("RESULTS_PATH", None)
    logloss = str(round(models.primary.metrics().get("weighted_log_loss", 0), 3))
    if results_path:
        res_path = f"{results_path}/{VERSION}_{logloss}"
        os.mkdir(res_path)
        with open(f"{res_path}/composite_model.pkl", "wb") as f:
            pickle.dump(models, f)

        with open(f"{res_path}/hyperparameters.pkl", "wb") as f:
            hp = {k: v.hyper_params for k, v in models.auxiliary.items()}
            hp["Primary"] = models.primary.hyper_params
            pickle.dump(hp, f)
        with open(f"{res_path}/summary.txt", "w") as f:
            f.write("\n".join(summary_lines))
        print(f"Model for {VERSION} saved")


import io

summary_lines = []


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    summary_lines.append(contents)


def print_model_info(models: CompositeModel):
    def print_data_stats(key, bundle):
        print("-" * 30)
        print(f"{key} Feature stats:")
        data__ = bundle.X[[col for col in bundle.X.columns if key in col]]
        print(data__.describe())
        print(data__.head())

    # for key, model in models.items():
    #     if key == 'Primary':
    #         continue
    #     print_data_stats(key, model.bundle)

    # print_data_stats('Primary', models['Primary'].bundle)
    def print_info(key, model):
        print_to_string("-" * 30)
        print_to_string(f"{key} Model Hyper Params:")
        print_to_string(pprint.pformat(model.hyper_params))
        print_to_string("-" * 30)
        print_to_string(f"{key} Model Metrics:")
        print_to_string(pprint.pformat(model.metrics()))

    for key, model in models.auxiliary.items():
        print_info(key, model)

    print_info("Primary", models.primary)


BEST_PARAMS = {
    "Primary": {
        "n_estimators": 191,
        "max_depth": 14,
        "learning_rate": 0.06220338628680403,
        "subsample": 0.5842204969997647,
        "colsample_bytree": 0.2214750920268831,
        "min_child_weight": 2,
        "gamma": 0.21240150099108568,
        "reg_alpha": 0.009232478046444677,
        "reg_lambda": 5.671364765595568e-05,
    },
    "Delta": {
        "n_estimators": 122,
        "max_depth": 15,
        "learning_rate": 0.07346740023932911,
        "subsample": 0.559195090518222,
        "colsample_bytree": 0.29361118426546196,
        "min_child_weight": 5,
        "gamma": 0.05808361216819946,
        "reg_alpha": 0.008661774840134775,
        "reg_lambda": 0.006011190005930914,
    },
    "Beta": {
        "n_estimators": 164,
        "max_depth": 5,
        "learning_rate": 0.09702107536403744,
        "subsample": 0.6994655844802531,
        "colsample_bytree": 0.3274034664069657,
        "min_child_weight": 6,
        "gamma": 0.18340450985343382,
        "reg_alpha": 0.0030424920053710816,
        "reg_lambda": 0.005247611840679216,
    },
    "Gamma": {
        "colsample_bytree": 0.4487470419390332,
        "gamma": 0.37641698570568116,
        "learning_rate": 0.09983459509160654,
        "max_depth": 6,
        "min_child_weight": 13,
        "n_estimators": 191,
        "reg_alpha": 0.00643785445276256,
        "reg_lambda": 0.0012425584014879776,
        "subsample": 0.7936767533192691,
    },
    "Alpha": {
        "colsample_bytree": 0.3432562109579994,
        "gamma": 0.12967210227795833,
        "learning_rate": 0.07735950868090753,
        "max_depth": 15,
        "min_child_weight": 7,
        "n_estimators": 99,
        "reg_alpha": 0.009769966554553772,
        "reg_lambda": 0.002178694812619288,
        "subsample": 0.49143481418652624,
    },
}


def train_primary_model(data: ICRData, hp_trials: int = 0) -> ModelResult:
    TARGET_COL = "Class"
    print(f"Training Primary Model")
    num_cols = get_num_cols(data.train)
    cat_cols = get_cat_cols(data.train)

    def preprocess_input() -> Bundle:
        print(f"Numeric Features: {num_cols}")

        print(f"Categorical Features: {cat_cols}")
        drop_cols = [
            col for col in data.train.columns if col not in cat_cols + num_cols
        ]
        X = data.train.drop(columns=drop_cols)
        print("All features:")
        print(X.columns)
        return Bundle(X=X, y=data.train[TARGET_COL], ids=data.train["Id"])

    trainer = GradientBoosterTrainer()
    bundle = preprocess_input()

    print(f"New num_cols {num_cols}")
    print(f"New cat_cols {cat_cols}")

    _, _, scale_pos_weight = get_scale_pos(bundle.y)
    print(cat_cols)
    print(num_cols)

    def binary_classifier(params) -> BinaryClassifier:
        return BinaryClassifier(
            bundle=bundle, params=params, scale_pos_weight=scale_pos_weight
        )

    def optimize_hp() -> Dict[str, Any]:
        optimizer = HyperparametersOptimizer(
            trainer=trainer, model_provider=binary_classifier
        )
        start = time.time()
        best_params = optimizer.optimize_params(
            bundle=bundle, n_splits=4, n_trials=hp_trials
        )
        print(
            f"It took {(time.time() - start) / 60} minutes to optimize params, {hp_trials} trials"
        )
        return best_params

    best_params = BEST_PARAMS["Primary"] if hp_trials <= 0 else optimize_hp()

    def train_predict_primary() -> ModelResult:
        print(f"Training, Predicting {VERSION}")
        # test_splits = round(bundle.positives / 25)
        test_splits = 5
        model = binary_classifier(best_params)

        model_result = trainer.train_model(
            model=model, bundle=bundle, n_splits=test_splits
        )
        print(f"Model {VERSION} Info:")

        return model_result

    return train_predict_primary()


# AUX_MODELS = ['Beta', 'Delta', 'Alpha', 'Gamma']


AUX_MODELS = ["Beta"]


def run_main(data: ICRData, hp_trials: int = 0, add_aux: bool = True) -> CompositeModel:
    start = time.time()
    auxiliary_dict = {}

    def add_feature_to_train(aux: str, data: ICRData) -> ICRData:
        if aux in {"Beta", "Delta"}:
            return add_feature_values(aux, data, auxiliary_dict[aux].bundle.label_map)
        elif aux in {"Alpha", "Gamma"}:
            data.train = merge_features_df(
                data.train, "Alpha", auxiliary_dict["Alpha"].result_df
            )
            return data
        else:
            raise ValueError(f"Unknown aux {aux}")

    if add_aux:
        for greek in AUX_MODELS:
            if greek == "Alpha":
                data = add_feature_to_train("Beta", data)
                data = add_feature_to_train("Delta", data)
                print_to_string(
                    f"Added Beta and Delta features values to dataset before training {greek} model"
                )
                print(
                    f"Added Beta and Delta features values to dataset before training {greek} model"
                )

            elif greek == "Gamma":
                data = add_feature_to_train("Alpha", data)
                print(
                    f"Added Alpha feature predictions to dataset before training {greek} model"
                )
                print_to_string(
                    f"Added Alpha feature predictions to dataset before training {greek} model"
                )

            print(f"Training {greek} auxiliary model")
            print_to_string(f"Training {greek} auxiliary model")
            auxiliary_dict[greek] = train_meta_model(
                greek,
                data=data,
                hp_trials=BEST_PARAMS[greek] if hp_trials <= 0 else hp_trials,
            )
            # data = add_feature_values(greek, data, auxiliary_dict[greek].bundle.label_map)
            # merge_features_df(data.train, greek, auxiliary_dict[greek].result_df)

    current_cols = data.train.columns
    for aux in auxiliary_dict.keys():

        def is_in_cols(aux):
            return len([col for col in current_cols if col.startswith(aux)]) > 0

        if not is_in_cols(aux):
            data = add_feature_to_train(aux, data)

    primary_model = train_primary_model(data, hp_trials)
    composite_model = CompositeModel(auxiliary=auxiliary_dict, primary=primary_model)

    print_model_info(composite_model)
    save_models_results(composite_model)
    print(
        f"ICR Conditions Model Run Completed. Overall Execution Took: {(time.time() - start) / 60} minutes"
    )
    print(
        f"Weighted Log Loss: {composite_model.primary.metrics().get('weighted_log_loss', None)} Version: {VERSION} "
    )
    print_to_string(
        f"Weighted Log Loss: {composite_model.primary.metrics().get('weighted_log_loss', None)}"
    )
    return composite_model


def predict(composite_model: CompositeModel):
    test = pd.read_csv(f"{base_path}/test.csv")

    def add_aux_to_test(key, model: ModelResult, test_df: pd.DataFrame) -> pd.DataFrame:
        print(f"predicting {key} on test")
        test_features = test_df.drop(columns=["Id"])
        aux_probs = []
        for pipeline in model.pipelines:
            pred_probs = pipeline.predict_proba(test_features)
            aux_probs.append(pred_probs)
        aux_probs = np.mean(np.array(aux_probs), axis=0)

        test_df = add_feature_predictions(
            test_df, key, aux_probs, model.bundle.reverse_labels_map()
        )
        return test_df

    for aux in AUX_MODELS:
        test = add_aux_to_test(aux, composite_model.auxiliary[aux], test)

    test_features = test.drop(columns=["Id"])

    print(test_features.columns)
    test_predictions = []
    primary_model = composite_model.primary
    for model in primary_model.pipelines:
        proba = model.predict_proba(
            test_features
        )  # picking the probabilities for both classes
        test_predictions.append(proba)

    # convert list of arrays to 3D array and then take mean along axis 0
    test_predictions = np.mean(np.array(test_predictions), axis=0)

    # Create a submission dataframe and save it to a csv file
    submission = pd.DataFrame(test_predictions, columns=["class_0", "class_1"])
    submission.insert(0, "Id", test["Id"])
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    # print(train.Class.value_counts())
    # b = preprocess_input(train)
    # print(train.dtypes)
    add_beta = True
    result = run_main(data=load_data(), hp_trials=10, add_aux=add_beta)
    predict(result)
    # print(result.metrics())
    # print(b.X.dtypes)
