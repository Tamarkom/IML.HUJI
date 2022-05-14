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
        best_thr = np.inf
        for index, feature in enumerate(X.T):
            thr_over_pos, thr_loss_pos = self._find_threshold(feature, y, 1)
            thr_over_neg, thr_loss_neg = self._find_threshold(feature, y, -1)
            if thr_loss_neg < thr_loss_pos and thr_loss_neg < best_thr:
                self.sign_ = -1
                self.threshold_ = thr_over_neg
                self.j_ = index
                best_thr = thr_loss_neg
            elif thr_loss_neg >= thr_loss_pos and thr_loss_pos < best_thr:
                self.sign_ = 1
                self.threshold_ = thr_over_pos
                self.j_ = index
                best_thr = thr_loss_pos

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
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

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
        best_thr_err = np.inf
        best_thr = values[0]
        for index, value in enumerate(values):
            pred = np.zeros(values.shape[0])
            pred[values >= value] = sign
            pred[values < value] = -sign
            thr_err = np.sum(np.abs(labels[np.sign(labels) != np.sign(pred)])) / values.shape[0]
            if thr_err < best_thr_err:
                best_thr_err = thr_err
                best_thr = value
        # check the max and min value
        pred = np.zeros(values.shape[0])
        pred[values >= np.max(values) + 1] = sign
        pred[values < np.max(values) + 1] = -sign
        thr_err = np.sum(np.abs(labels[np.sign(labels) != np.sign(pred)])) / values.shape[0]
        if thr_err < best_thr_err:
            best_thr_err = thr_err
            best_thr = np.max(values) + 1
        pred = np.zeros(values.shape[0])
        pred[values >= np.min(values) - 1] = sign
        pred[values < np.min(values) - 1] = -sign
        thr_err = np.sum(np.abs(labels[np.sign(labels) != np.sign(pred)])) / values.shape[0]
        if thr_err < best_thr_err:
            best_thr_err = thr_err
            best_thr = np.min(values) - 1
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
        return np.sum(np.abs(y[np.sign(y) != np.sign(self.predict(X))])) / X.shape[0]