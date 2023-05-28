import datetime
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class ICRData:
    train: pd.DataFrame
    greeks: pd.DataFrame


def get_num_cols(X) -> List[str]:
    return X.select_dtypes(include=["float64", "float32"]).columns.tolist()


def get_cat_cols(X) -> List[str]:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if "Id" in cat_cols:
        cat_cols.remove("Id")
    return cat_cols


@dataclass
class Bundle:
    X: pd.DataFrame
    y: pd.Series
    ids: pd.DataFrame
    label_map: Optional[OrderedDict[str, int]] = None

    def get_num_cols(self) -> List[str]:
        return get_num_cols(self.X)

    def get_cat_cols(self) -> List[str]:
        return get_cat_cols(self.X)

    def reverse_labels_map(self) -> Dict[int, str]:
        return {v: k for k, v in self.label_map.items()}


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    ids_test: List


@dataclass
class ModelResult:
    bundle: Bundle
    model_info: Dict[str, Any]
    result_df: pd.DataFrame
    hyper_params: Dict[str, Any]
    pipelines: List[Pipeline]

    def metrics(self) -> Dict[str, float]:
        return self.model_info.get("metrics", {})

    def get_trials(self) -> int:
        return self.model_info.get("additional", {}).get("study_trials", -1)

    def set_trials(self, n_trails: int):
        self.model_info.setdefault("additional", {})["study_trials"] = n_trails


MODEL_TYPE = "xgboost"
VERSION = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
