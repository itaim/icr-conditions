import os
import pickle
import pprint
import time
from typing import Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from auxiliary_models import add_feature_cols, add_meta_feature, train_meta_model
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


def run_main(
    data: ICRData, hp_trials: int = 0, add_aux: bool = True
) -> Dict[str, ModelResult]:
    TARGET_COL = "Class"
    beta_model = None
    delta_model = None
    alpha_model = None
    # gamma_model = None
    if add_aux:
        # data_with_beta = add_meta_feature('Beta',data,beta_model.bundle.label_map)
        delta_model = train_meta_model("Delta", data=data, hp_trials=0)
        data = add_meta_feature("Delta", data, delta_model.bundle.label_map)
        beta_model = train_meta_model("Beta", data=data, hp_trials=0)
        data = add_meta_feature("Beta", data, beta_model.bundle.label_map)
        # gamma_model = train_meta_model('Gamma',data=data,hp_trials=50)
        alpha_model = train_meta_model("Alpha", data=data, hp_trials=0)
        # data.train = data.train.merge(gamma_model.result_df.drop(columns=['target']), on=['Id'], how='inner')
        data.train = data.train.merge(
            alpha_model.result_df.drop(columns=["target"]), on=["Id"], how="inner"
        )

        # data = add_meta_feature('Alpha', data, alpha_model.bundle.label_map,probabilities=alpha_model.result_df)

    #     try rebuilding
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

    def create_model(params) -> BinaryClassifier:
        return BinaryClassifier(
            bundle=bundle, params=params, scale_pos_weight=scale_pos_weight
        )

    if hp_trials > 0:
        optimizer = HyperparametersOptimizer(
            trainer=trainer, model_provider=create_model
        )
        start = time.time()
        best_params = optimizer.optimize_params(
            bundle=bundle, n_splits=4, n_trials=hp_trials
        )
        print(
            f"It took {(time.time() - start) / 60} minutes to optimize params, {hp_trials} trials"
        )
    else:
        best_params = {
            "n_estimators": 191,
            "max_depth": 14,
            "learning_rate": 0.06220338628680403,
            "subsample": 0.5842204969997647,
            "colsample_bytree": 0.2214750920268831,
            "min_child_weight": 2,
            "gamma": 0.21240150099108568,
            "reg_alpha": 0.009232478046444677,
            "reg_lambda": 5.671364765595568e-05,
        }

    start = time.time()
    print(f"Training, Predicting {VERSION}")
    # test_splits = round(bundle.positives / 25)
    test_splits = 5
    model = create_model(best_params)
    if add_aux:
        print(f"Beta")
        beta__ = bundle.X[[col for col in bundle.X.columns if "Beta" in col]]
        print(beta__.describe())
        print(beta__.head())
        print(f"Delta")
        delta__ = bundle.X[[col for col in bundle.X.columns if "Delta" in col]]
        print(delta__.describe())
        print(delta__.head())
    model_result = trainer.train_model(model=model, bundle=bundle, n_splits=test_splits)
    print(f"Model {VERSION} Info:")
    if add_aux:
        print("-" * 30)
        print(f"Beta Model Hyper Params:")
        pprint.pp(beta_model.hyper_params)
        print("-" * 30)
        print(f"Beta Model Metrics:")
        pprint.pp(beta_model.metrics())
        print("-" * 30)
        print(f"Delta Model Hyper Params:")
        pprint.pp(delta_model.hyper_params)
        print("-" * 30)
        print(f"Delta Model Metrics:")
        pprint.pp(delta_model.metrics())
        print("-" * 30)
        print(f"Alpha Model Hyper Params:")
        pprint.pp(alpha_model.hyper_params)
        print("-" * 30)
        print(f"Alpha Model Metrics:")
        pprint.pp(alpha_model.metrics())
        print("-" * 30)
        # print(f'Gamma Model Hyper Params:')
        # pprint.pp(gamma_model.hyper_params)
        # print('-' * 30)
        # print(f'Gamma Model Metrics:')
        # pprint.pp(gamma_model.metrics())
        # print('-' * 30)

    print("-" * 30)
    print(f"Main Model Hyper Params:")
    pprint.pp(model_result.hyper_params)
    print("-" * 30)
    print(f"Main Model Metrics:")
    pprint.pp(model_result.metrics())
    print(
        f"Weighted Log Loss : {model_result.metrics().get('weighted_log_loss', None)}"
    )
    print("-" * 30)
    results_path = os.environ.get("RESULTS_PATH", None)
    if results_path:
        res_path = f"{results_path}/{VERSION}"
        os.mkdir(res_path)
        pd.DataFrame.from_records([model_result.model_info]).to_csv(
            f"{res_path}/model_info.csv"
        )
        model_result.result_df.to_csv(f"{res_path}/results.csv")
        with open(f"{res_path}/pipeline.pkl", "wb") as f:
            pickle.dump(model_result.pipelines, f)

    print(f"Model for {VERSION} saved")
    print(
        f"ICR Conditions Model Run Completed. Overall Execution Took: {(time.time() - start) / 60} minutes"
    )
    return {"beta": beta_model, "primary": model_result}


def predict(models: Dict[str, ModelResult], add_beta: bool):
    test = pd.read_csv(f"{base_path}/test.csv")

    def add_beta_to_test() -> pd.DataFrame:
        beta_model = models["beta"]
        # predict and add beta features to test df
        test_features = test.drop(columns=["Id"])
        beta_probs = []
        for model in beta_model.pipelines:
            pred_probs = model.predict_proba(test_features)
            beta_probs.append(pred_probs)
        beta_probs = np.mean(np.array(beta_probs), axis=0)
        test_features = add_feature_cols(
            test, "Beta", beta_probs, beta_model.bundle.reverse_labels_map()
        ).drop(columns=["Id"])
        # print(test_features[['Beta_0', 'Beta_1', 'Beta_2']].describe())
        return test_features

    if add_beta:
        test_features = add_beta_to_test()
    else:
        test_features = test.drop(columns=["Id"])

    print(test_features.columns)
    test_predictions = []
    model_result = models["primary"]
    for model in model_result.pipelines:
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
    result = run_main(data=load_data(), hp_trials=0, add_aux=add_beta)
    # predict(result, add_beta=add_beta)
    # print(result.metrics())
    # print(b.X.dtypes)
