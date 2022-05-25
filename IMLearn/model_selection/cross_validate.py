from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator
    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data
    X: ndarray of shape (n_samples, n_features)
       Input data to fit
    y: ndarray of shape (n_samples, )
       Responses of input data to fit to
    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.
    cv: int
        Specify the number of folds.
    Returns
    -------
    train_score: float
        Average train score over folds
    validation_score: float
        Average validation score over folds
    """
    # separate to cv equally sized set
    separated_x = np.array_split(X, cv, axis=0)
    separated_y = np.array_split(y, cv, axis=0)
    train_score = 0
    test_score = 0
    if cv == 0:
        return 0, 0
    for k in range(cv):
        # train the model for all data except s_k
        # TODO: improve
        test_x, test_y = separated_x[k], separated_y[k]
        train_x = np.concatenate([separated_x[i] for i in range(cv) if i != k])
        train_y = np.concatenate([separated_y[i] for i in range(cv) if i != k])
        estimator.fit(train_x, train_y)

        # calculate the error for s_k
        # TODO: what is the problem
        train_score += scoring(train_y, estimator.predict(train_x))
        test_score += scoring(test_y, estimator.predict(test_x))
    return train_score / cv, test_score / cv
