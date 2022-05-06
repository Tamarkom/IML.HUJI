from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm
    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split
    self.j_ : int
        The index of the feature by which to split the data
    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_thr = 1
        for index, feature in enumerate(X.T):
            thr_over_pos, thr_loss_pos = self._find_threshold(feature, y, 1)
            thr_over_neg, thr_loss_neg = self._find_threshold(feature, y, -1)
            if thr_loss_neg < thr_loss_pos and thr_loss_neg < best_thr:
                self.sign_ = -1
                best_thr = thr_over_neg
                self.j_ = index
            elif thr_loss_neg > thr_loss_pos and thr_loss_pos < best_thr:
                self.sign_ = 1
                best_thr = thr_over_pos
                self.j_ = index

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.sign_ if X[self.j_] > self.threshold_ else -self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature
        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for
        labels: ndarray of shape (n_samples,)
            The labels to compare against
        sign: int
            Predicted label assigned to values equal to or above threshold
        Returns
        -------
        thr: float
            Threshold by which to perform split
        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold
        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        index_sort_by_x = np.argsort(values)
        x = values[index_sort_by_x]
        y = labels[index_sort_by_x]
        best_thr_err = 1
        best_thr = values[0]
        for index, value in enumerate(x):
            pred = np.zeros(x.shape[0])
            pred[x > value] = sign
            pred[x <= value] = -sign
            thr_err = misclassification_error(y, pred)
            if thr_err < best_thr_err:
                best_thr_err = thr_err
                best_thr = value
        return best_thr, best_thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))