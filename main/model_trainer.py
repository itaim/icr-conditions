import pprint
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Any, Dict
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from metrics import SCORERS, OPTIMIZATION_OBJECTIVES, get_scale_pos

# from main.main import num_cols, cat_cols
from model import Bundle, ModelResult, MODEL_TYPE, Split, VERSION


class Model(ABC):
    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def num_cols(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def cat_cols(self) -> List[str]:
        pass

    @abstractmethod
    def classifier(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self, split: Split):
        raise NotImplementedError()

    @abstractmethod
    def complete(self) -> ModelResult:
        raise NotImplementedError()

    def prepare_eval_X(self, split: Split):
        eval_feature_transformer = self.build_feature_transformer()
        eval_feature_transformer.fit(split.X_train, split.y_train)
        X_eval = eval_feature_transformer.transform(split.X_test)
        return X_eval

    def build_feature_transformer(self) -> ColumnTransformer:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        if MODEL_TYPE == "xgboost":
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
        elif MODEL_TYPE == "lightgbm":
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                ]
            )
        else:
            raise ValueError(f"{MODEL_TYPE} must be xgboost or lightgbm")

        return ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.num_cols),
                ("cat", categorical_transformer, self.cat_cols),
            ]
        )

    def build_pipeline(self, upsample: bool = True) -> Pipeline:
        if upsample:
            return ImbPipeline(
                [
                    (
                        "preprocessor",
                        self.build_feature_transformer(),
                    ),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", self.classifier()),
                ]
            )
        else:
            return Pipeline(
                [
                    (
                        "preprocessor",
                        self.build_feature_transformer(),
                    ),
                    ("classifier", self.classifier()),
                ]
            )


class BinaryClassifier(Model):
    def __init__(self, bundle: Bundle, params: Dict[str, Any], scale_pos_weight: float):
        self.bundle = bundle
        self._cat_cols = bundle.get_cat_cols()
        self._num_cols = bundle.get_num_cols()
        self._params = params
        self._scale_pos_weight = scale_pos_weight
        self.pipelines = []
        self.metrics = defaultdict(list)
        self.ids_list = []
        self.predictions = []
        self.targets = []

    @property
    def cat_cols(self) -> List[str]:
        return self._cat_cols

    @property
    def num_cols(self) -> List[str]:
        return self._num_cols

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def classifier(self):
        if MODEL_TYPE == "xgboost":
            if self.params:
                #
                clf = xgb.XGBClassifier(
                    objective="binary:logistic",
                    tree_method="hist",
                    enable_categorical=True,
                    early_stopping_rounds=20,
                    eval_metric=["logloss"],
                    scale_pos_weight=self._scale_pos_weight,
                    **self.params,
                )
            else:
                clf = xgb.XGBClassifier(
                    objective="binary:logistic",
                    tree_method="hist",
                    enable_categorical=True,
                    scale_pos_weight=self._scale_pos_weight,
                )
        elif MODEL_TYPE == "lightgbm":
            clf = lgb.LGBMClassifier()
            if self.params:
                clf.set_params(**self.params)
        else:
            raise NotImplementedError(f"unknown model type {MODEL_TYPE}")
        return clf

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
        y_pred_proba = pipeline.predict_proba(split.X_test)[:, 1]
        self.score(split.y_test, y_pred_proba)
        self.predictions.extend(y_pred_proba)
        self.targets.extend(split.y_test.values.tolist())
        self.ids_list.extend(split.ids_test)

    def score(self, y_test, y_pred_proba):
        y_pred = y_pred_proba > 0.5

        for metric_name, (fscore, needs_proba) in SCORERS.items():
            self.metrics[metric_name].append(
                fscore(y_test, y_pred_proba if needs_proba else y_pred)
            )
        #     "youden_index": youdens_index(y, predictions),
        #             "max_gmeans_roc": max_roc_distance(y, predictions),
        return y_pred_proba

    def complete(self) -> ModelResult:
        result_df = pd.DataFrame(data={"id": self.ids_list, "target": self.targets})
        result_df["prediction"] = self.predictions

        metrics = {k: np.mean(v) for k, v in self.metrics.items()}

        info_record = {
            "metrics": metrics,
        }
        additional = {
            "model_type": MODEL_TYPE,
            "optimization_objectives": [obj for _, obj in OPTIMIZATION_OBJECTIVES],
            "hyper_prams": self.params.copy() if self.params else {},
            "model_version": VERSION,
        }

        negative, positive, scale_pos_weight = get_scale_pos(np.asarray(self.targets))
        info_record["class_0"] = negative
        info_record["class_1"] = positive
        info_record["scale_pos_weight"] = scale_pos_weight
        print(f"Negative/Positive: {scale_pos_weight}")
        info_record["additional"] = additional

        return ModelResult(
            bundle=self.bundle,
            model_info=info_record,
            result_df=result_df,
            hyper_params=self.params,
            pipelines=self.pipelines,
        )


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(self, model: Model, bundle: Bundle, n_splits: int) -> ModelResult:
        raise NotImplementedError()


class GradientBoosterTrainer(ModelTrainer):
    def train_model(self, model: Model, bundle: Bundle, n_splits: int) -> ModelResult:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = bundle.X
        y = bundle.y
        ids_list = []
        print(f"Starting evaluation on {n_splits} splits")
        for train_index, test_index in cv.split(X, y):
            print(f"Fitting model on split: {len(ids_list)}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            split = Split(
                X_train, y_train, X_test, y_test, bundle.ids.values[test_index]
            )
            model.train(split)

        return model.complete()
