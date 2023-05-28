import pprint
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable

import optuna
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

from metrics import OPTIMIZATION_OBJECTIVES
from model import MODEL_TYPE, VERSION
from model_trainer import GradientBoosterTrainer, ModelTrainer, Bundle


class TrialSorter(ABC):
    @abstractmethod
    def aggregate_metrics(self, x: FrozenTrial) -> float:
        pass


class DefaultTrialSorter(TrialSorter):
    def __init__(self, optimization_objectives):
        self.directions, self.objectives = zip(*optimization_objectives)

    def aggregate_metrics(self, x: FrozenTrial) -> float:
        print(f"Trial {x.number} - values: {x.values}")
        return sum(
            [
                v if self.directions[i] == "maximize" else -v
                for i, v in enumerate(x.values)
                if i < 2
            ]
        )


class SingleMetricSorter(TrialSorter):
    def __init__(self, optimization_objectives, metric: str):
        self.directions, self.objectives = zip(*optimization_objectives)
        self.value_index = self.objectives.index(metric)

    def aggregate_metrics(self, x: FrozenTrial) -> float:
        print(f"Trial {x.number} - values: {x.values}")
        metric_val = x.values[self.value_index]
        return (
            metric_val
            if self.directions[self.value_index] == "maximize"
            else -metric_val
        )


class HyperparametersOptimizer:
    def __init__(self, trainer: GradientBoosterTrainer, model_provider: Callable):
        self.trainer: ModelTrainer = trainer
        self.model_provider = model_provider

    def objective(self, trial, bundle: Bundle, n_splits: int, objectives: List[str]):
        if MODEL_TYPE == "xgboost":
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 75, 200),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
                "subsample": trial.suggest_float("subsample", 0.2, 0.8),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 1e-2),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 1e-2),
            }

        elif MODEL_TYPE == "lightgbm":
            param = {
                "classifier__n_estimators": trial.suggest_int("n_estimators", 35, 150),
                "classifier__max_depth": trial.suggest_int("max_depth", 5, 12),
                "classifier__learning_rate": trial.suggest_uniform(
                    "learning_rate", 0.001, 0.1
                ),
                "classifier__subsample": trial.suggest_uniform("subsample", 0.2, 0.8),
                "classifier__colsample_bytree": trial.suggest_uniform(
                    "colsample_bytree", 0.2, 0.9
                ),
                "classifier__min_child_weight": trial.suggest_int(
                    "min_child_weight", 1, 7
                ),
                "classifier__num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "classifier__reg_alpha": trial.suggest_uniform("reg_alpha", 0, 1),
                "classifier__reg_lambda": trial.suggest_uniform("reg_lambda", 0, 1),
                "classifier__feature_fraction": trial.suggest_uniform(
                    "feature_fraction", 0.4, 1.0
                ),
                "classifier__bagging_fraction": trial.suggest_uniform(
                    "bagging_fraction", 0.4, 1.0
                ),
                "classifier__min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 100
                ),
            }

        else:
            raise ValueError("Invalid model_type. Must be 'xgboost' or 'lightgbm'.")

        model = self.model_provider(param)
        # Perform cross-validation and take the mean of the scores
        result = self.trainer.train_model(model=model, bundle=bundle, n_splits=n_splits)
        metrics = result.metrics()
        scores = [metrics[objective] for objective in objectives]
        print(f"Trial {trial.number} CV {objectives}: {scores}")
        return tuple(scores)

    def optimize_params(
        self,
        bundle: Bundle,
        n_splits: int,
        n_trials=500,
        optimization_objectives: List[Tuple[str, str]] = OPTIMIZATION_OBJECTIVES,
        sorter: TrialSorter = SingleMetricSorter(
            OPTIMIZATION_OBJECTIVES, "weighted_log_loss"
        ),
    ) -> Dict[str, Any]:
        sampler = TPESampler(seed=42)
        print(f"Objectives {optimization_objectives}")
        directions, objectives = zip(*optimization_objectives)
        study = optuna.create_study(
            directions=directions,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            study_name=f"icr-conditions-{VERSION}",
        )
        print(
            f"Starting hyperparameters search. {n_trials} trials, {n_splits} cv splits"
        )
        start = time.time()
        study.optimize(
            lambda trial: self.objective(trial, bundle, n_splits, objectives),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        print(f"Pareto Front")

        best_trial = sorted(
            study.best_trials, key=lambda x: sorter.aggregate_metrics(x)
        )[-1]

        print(f"Hyperparameters search took {(time.time() - start) / 60} minutes.")
        print(f"Best values: {pprint.pformat(best_trial.values)}")
        print(f"Best hyper-parameters: {pprint.pformat(best_trial.params)}")
        return best_trial.params


def find_by_randomized_cv_search(
    X_train,
    y_train,
    pipeline,
    scoring: str = "roc_auc",
    model_type: str = "xgboost",
    n_iter=500,
):
    if model_type == "xgboost":
        param_dist = {
            "classifier__n_estimators": randint(35, 200),
            "classifier__max_depth": randint(5, 12),
            "classifier__learning_rate": uniform(0.001, 0.1),
            "classifier__subsample": uniform(0.2, 0.8),
            "classifier__colsample_bytree": uniform(0.2, 0.8),
            "classifier__min_child_weight": randint(1, 30),
            "classifier__gamma": uniform(0, 1),
        }
    elif model_type == "lightgbm":
        param_dist = {
            "classifier__n_estimators": randint(50, 300),
            "classifier__max_depth": randint(3, 10),
            "classifier__learning_rate": uniform(0.01, 0.3),
            "classifier__subsample": uniform(0.5, 0.5),
            "classifier__colsample_bytree": uniform(0.5, 0.5),
            "classifier__min_child_weight": randint(1, 7),
            "classifier__num_leaves": randint(20, 100),
            "classifier__reg_alpha": uniform(0, 1),
            "classifier__reg_lambda": uniform(0, 1),
        }
    else:
        raise ValueError("Invalid model_type. Must be 'xgboost' or 'lightgbm'.")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_params = {
        key.replace("classifier__", ""): value for key, value in best_params.items()
    }
    print(f"Best parameters: {pprint.pformat(best_params)}")
    return best_params
