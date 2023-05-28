from typing import Tuple

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    log_loss,
    roc_curve,
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
)
from sklearn.utils import compute_sample_weight


def weighted_matthews_corrcoef(y_true, y_pred) -> float:
    return matthews_corrcoef(
        y_true, y_pred, sample_weight=compute_sample_weight("balanced", y_true)
    )


def balanced_precision_recall(y_true, y_pred) -> float:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return ((precision + recall) - abs(precision - recall)) / 2


def weighted_log_loss(y_true, y_pred, labels=None) -> float:
    print(labels)
    return log_loss(
        y_true,
        y_pred,
        sample_weight=compute_sample_weight("balanced", y_true),
        labels=labels,
    )


def balanced_weighted_log_loss(y_true, y_pred, eps=1e-15):
    # Compute class prevalences
    prevalence_0 = np.mean(1 - y_true)
    prevalence_1 = np.mean(y_true)

    # Compute weights
    weight_0 = 1 / prevalence_0
    weight_1 = 1 / prevalence_1

    # Cap predicted probabilities
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Compute log loss for each class
    # np.mean() function is taking care of dividing by N0 and N1
    log_loss_0 = -np.mean((1 - y_true) * np.log(1 - y_pred))
    log_loss_1 = -np.mean(y_true * np.log(y_pred))

    # Combine the log losses
    total_log_loss = weight_0 * log_loss_0 + weight_1 * log_loss_1

    return float(total_log_loss / (weight_0 + weight_1))


def balance_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # y_pred / np.sum(y_pred, axis=1)[:, None]
    nc = np.bincount(y_true)
    w0, w1 = 1 / (nc[0] / y_true.shape[0]), 1 / (nc[1] / y_true.shape[0])

    logloss = (
        -w0 / nc[0] * (np.sum(np.where(y_true == 0, 1, 0) * np.log(y_pred[:, 0])))
        - w1 / nc[1] * (np.sum(np.where(y_true != 0, 1, 0) * np.log(y_pred[:, 1])))
    ) / (w0 + w1)

    return logloss


def youdens_index(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youdens_index = tpr - fpr
    return float(thresholds[np.argmax(youdens_index)])


def max_roc_distance(y_true, y_score):
    # maximizes the distance to the line of no-discrimination (diagonal line) in the ROC space.
    # Calculation maximizes the geometric mean of sensitivity and specificity.
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    return float(thresholds[ix])


def get_scale_pos(y) -> Tuple[int, int, float]:
    negative = int((y == 0).sum())
    positive = int((y == 1).sum())
    return negative, positive, negative / positive


SCORERS = {
    "roc_auc_score": (roc_auc_score, True),
    "recall_score": (recall_score, False),
    "precision_score": (precision_score, False),
    "f1_score": (f1_score, False),
    "log_loss": (log_loss, True),
    "weighted_log_loss": (weighted_log_loss, True),
    # "balanced_weighted_log_loss": (balanced_weighted_log_loss, True),
    "accuracy_score": (accuracy_score, False),
    "matthews_corrcoef": (weighted_matthews_corrcoef, False),
    "balanced_precision_recall": (balanced_precision_recall, False),
    "average_precision_score": (average_precision_score, True),
}
OPTIMIZATION_OBJECTIVES = [
    ("minimize", "weighted_log_loss"),
    ("maximize", "matthews_corrcoef"),
]
SCORING = OPTIMIZATION_OBJECTIVES[0][1]
