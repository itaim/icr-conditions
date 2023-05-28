import numpy as np
from main.metrics import balanced_weighted_log_loss, balance_logloss, weighted_log_loss


def test_balance_logloss():
    # Define some test cases
    test_cases = [
        {
            "y_true": np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0]),
            "y_pred": np.array(
                [
                    [0.1, 0.9],
                    [0.8, 0.2],
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.1, 0.9],
                    [0.3, 0.7],
                    [0.6, 0.4],
                    [0.7, 0.3],
                    [0.1, 0.9],
                    [0.6, 0.4],
                ]
            ),
        },
        {
            "y_true": np.array([1, 1, 1, 1, 0, 0, 0, 0]),
            "y_pred": np.array(
                [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.2, 0.8],
                    [0.1, 0.9],
                    [0.6, 0.4],
                    [0.7, 0.3],
                    [0.7, 0.3],
                    [0.6, 0.4],
                ]
            ),
        },
        {"y_true": np.array([1, 0]), "y_pred": np.array([[0.3, 0.7], [0.8, 0.2]])},
    ]

    # Evaluate each test case
    for i, test_case in enumerate(test_cases):
        y_true = test_case["y_true"]
        y_pred = test_case["y_pred"]
        logloss_1 = balance_logloss(y_true, y_pred)
        logloss_2 = balanced_weighted_log_loss(y_true, y_pred[:, 1])
        logloss_3 = weighted_log_loss(y_true, y_pred[:, 1])
        print(f"Test case {i+1}:")
        print(f"balance_logloss: {logloss_1}")
        print(f"balanced_weighted_log_loss: {logloss_2}")
        print(f"weighted_log_loss: {logloss_3}")
        print()


# Run the test
# test_balance_logloss(balance_logloss, balanced_weighted_log_loss)
