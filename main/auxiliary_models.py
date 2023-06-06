import time
from collections import defaultdict, OrderedDict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    average_precision_score,
    precision_recall_curve,
)

from hyperparameters import (
    HyperparametersOptimizer,
    AggregatedObjectivesUtility,
    WeightedComplexityUtility,
)
from metrics import weighted_log_loss
from model import (
    Bundle,
    ModelResult,
    Split,
    MODEL_TYPE,
    VERSION,
    ICRData,
    get_num_cols,
    get_cat_cols,
)
from model_trainer import GradientBoosterTrainer, Model


def get_bundle(data: ICRData, greek: str):
    df = data.train.drop(columns=["Class"]).merge(
        data.greeks[["Id", greek]], on=["Id"], how="inner"
    )
    labels = df[greek].unique()
    label_map = OrderedDict((l, i) for i, l in enumerate(labels))

    num_cols = get_num_cols(data.train)
    cat_cols = get_cat_cols(data.train)
    print(f"Numeric Features: {num_cols}")
    print(f"Categorical Features: {cat_cols}")

    drop_cols = [col for col in df.columns if col not in cat_cols + num_cols]
    X = df.drop(columns=drop_cols)
    print(df[greek].value_counts())
    return Bundle(X=X, y=df[greek].map(label_map), ids=df["Id"], label_map=label_map)


AUX_EXCLUDE_COLS = {
    "Alpha": {"Alpha_A"},
    "Gamma": {"Gamma_M", "Gamma_N"},
}


def merge_features_df(
    df: pd.DataFrame, prefix: str, features: pd.DataFrame
) -> pd.DataFrame:
    cols = [
        col for col in features.columns if col not in AUX_EXCLUDE_COLS.get(prefix, {})
    ]
    selected_features_df = features[cols]
    print(
        f"merge_features_df - Merging {prefix} prediction. Cols {selected_features_df.columns}"
    )
    return df.merge(selected_features_df, on=["Id"], how="inner")


def add_feature_predictions(
    df, prefix: str, predictions, label_map: Dict[int, str]
) -> pd.DataFrame:
    probabilities_df = pd.DataFrame(
        predictions,
        columns=[f"{prefix}_{label_map[i]}" for i in range(len(predictions[0]))],
    )
    exclude = AUX_EXCLUDE_COLS.get(prefix, None)
    if exclude:
        probabilities_df.drop(columns=exclude, inplace=True)
    # print(f'add_feature_predictions {prefix} prediction. probabilities_df {probabilities_df.columns}')
    return pd.concat([df, probabilities_df], axis=1)


def add_feature_values(greek: str, data: ICRData, lm: OrderedDict[str, int]) -> ICRData:
    def map_to_probs(value):
        prob_array = [0.0] * len(lm)
        prob_array[lm[value]] = 1.0
        return prob_array

    probabilities = data.greeks[greek].apply(map_to_probs)
    probabilities_df = pd.DataFrame(
        probabilities.tolist(), columns=[f"{greek}_{name}" for name in lm.keys()]
    )
    new_train = pd.concat([data.train, probabilities_df], axis=1)
    print(f"add_feature_values {greek} prediction. New train cols: {new_train.columns}")
    return ICRData(train=new_train, greeks=data.greeks)


class MultiClassClassifier(Model):
    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def num_cols(self) -> List[str]:
        return self._num_cols

    @property
    def cat_cols(self) -> List[str]:
        return self._cat_cols

    def __init__(
        self,
        target: str,
        bundle: Bundle,
        params: Dict[str, Any],
        optimization_objectives: List[Tuple[str, str]],
    ):
        self._name = target
        self.optimization_objectives = optimization_objectives
        self.bundle = bundle
        self._cat_cols = bundle.get_cat_cols()
        self._num_cols = bundle.get_num_cols()
        self._params = params
        self.pipelines = []
        self.metrics = defaultdict(list)
        self.ids_list = []
        self.predictions = []
        self.targets = []

    def classifier(self):
        num_class = len(self.bundle.label_map)
        if self.params:
            return xgb.XGBClassifier(
                objective="multi:softmax",
                num_class=num_class,
                # missing=1,
                early_stopping_rounds=10,
                eval_metric=["merror", "mlogloss"],
                seed=42,
                **self.params,
            )
        else:
            return xgb.XGBClassifier(
                objective="multi:softmax",
                num_class=num_class,
                # missing=1,
                gamma=0,  # default gamma value
                learning_rate=0.1,
                max_depth=3,
                reg_lambda=1,  # default L2 value
                subsample=1,  # default subsample value
                colsample_bytree=1,  # default colsample_bytree value
                early_stopping_rounds=10,
                eval_metric=["merror", "mlogloss"],
                seed=42,
            )

    def train(self, split: Split):
        pipeline = self.build_pipeline()
        pipeline.fit(
            split.X_train,
            split.y_train,
            classifier__eval_set=[(self.prepare_eval_X(split), split.y_test)],
            classifier__verbose=False,
        )
        best_iteration = pipeline["classifier"].best_iteration
        best_score = pipeline["classifier"].best_score
        print(f"best iter {best_iteration}. best score {best_score}")
        self.pipelines.append(pipeline)
        y_pred_proba = pipeline.predict_proba(split.X_test)
        self.score(split.y_test, y_pred_proba)
        self.predictions.extend(y_pred_proba)
        self.targets.extend(split.y_test.values.tolist())
        self.ids_list.extend(split.ids_test)

    def average_precision_score(self, Y_test, y_score, n_classes):
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                Y_test[:, i], y_score[:, i]
            )
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            Y_test.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(
            Y_test, y_score, average="micro"
        )
        print(
            "Average precision score, micro-averaged over all classes: {0:0.2f}".format(
                average_precision["micro"]
            )
        )
        return average_precision["micro"]

    def score(self, y_test, y_pred_proba):
        y_pred = np.argmax(y_pred_proba, axis=1)

        self.metrics["accuracy_score"].append(accuracy_score(y_test, y_pred))
        self.metrics["confusion_matrix"].append(confusion_matrix(y_test, y_pred))
        # metrics['classification_report'].append(classification_report(y_test, y_pred_proba))
        # self.metrics['average_precision_score'].append(
        #     self.average_precision_score(y_test, y_pred_proba, len(self.bundle.label_map)))
        self.metrics["weighted_log_loss"].append(
            weighted_log_loss(
                y_test, y_pred_proba, labels=list(self.bundle.label_map.values())
            )
        )
        # self.metrics['roc_auc_score'].append(roc_auc_score(y_test, y_pred, multi_class='ovr'))
        # self.metrics['roc_auc_score'].append(roc_auc_score(y_test, y_pred, average='weighted'))
        self.metrics["matthews_corrcoef"].append(matthews_corrcoef(y_test, y_pred))

    def complete(self) -> ModelResult:
        result_df = pd.DataFrame(data={"Id": self.ids_list, self.name: self.targets})
        result_df = add_feature_predictions(
            result_df, self.name, self.predictions, self.bundle.reverse_labels_map()
        )
        metrics = {k: np.mean(v) for k, v in self.metrics.items()}

        info_record = {
            "metrics": metrics,
        }
        additional = {
            "model_type": MODEL_TYPE,
            "optimization_objectives": [obj for _, obj in self.optimization_objectives],
            "hyper_prams": self.params.copy() if self.params else {},
            "model_version": VERSION,
        }

        info_record["additional"] = additional
        return ModelResult(
            name=self.name,
            bundle=self.bundle,
            model_info=info_record,
            result_df=result_df,
            hyper_params=self.params,
            pipelines=self.pipelines,
        )


def train_meta_model(
    greek: str, data: ICRData, hp_trials: int | Dict[str, Any] = 100, n_splits: int = 5
) -> ModelResult:
    bundle = get_bundle(data=data, greek=greek)
    print(f"Training meta model {greek}")
    trainer = GradientBoosterTrainer()

    # num_cols=num_cols, cat_cols=cat_cols, clf=xgb_multiclass,
    #                                      scorer=MultiClassScorer(list(bundle.label_map.values()))
    # optimization_objectives = [('maximize', 'accuracy_score')]

    # optimization_objectives = [('maximize', 'accuracy_score'),('maximize', 'accuracy_score'), ('minimize', 'weighted_log_loss')]

    def create_model(params) -> MultiClassClassifier:
        return MultiClassClassifier(
            target=greek,
            bundle=bundle,
            params=params,
            optimization_objectives=optimization_objectives,
        )

    optimizer = HyperparametersOptimizer(trainer=trainer, model_provider=create_model)

    if isinstance(hp_trials, int):
        optimization_objectives = [
            ("maximize", "matthews_corrcoef"),
            ("minimize", "weighted_log_loss"),
        ]
        aux_objectives_utility = AggregatedObjectivesUtility(optimization_objectives)
        aux_optimization_utility = WeightedComplexityUtility(
            optimization_objectives, aux_objectives_utility, complexity_factor=0.0001
        )

        start = time.time()
        best_params = optimizer.optimize_params(
            bundle=bundle,
            n_splits=4,
            n_trials=hp_trials,
            utility=aux_optimization_utility,
        )
        print(
            f"It took {(time.time() - start) / 60} minutes to optimize params, {hp_trials} trials"
        )
    else:
        best_params = hp_trials

    model = create_model(best_params)
    return trainer.train_model(model=model, bundle=bundle, n_splits=n_splits)
