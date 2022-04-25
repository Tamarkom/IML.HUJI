from typing import NoReturn

import pandas as pd

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        df = pd.concat([pd.DataFrame(y), pd.DataFrame(X)], axis=1, ignore_index=True)
        self.mu_ = df.groupby(by=0).mean().to_numpy()
        sigma = 0
        for i in range(len(X)):
            sigma += np.outer(X[i] - self.mu_[y[i]], X[i] - self.mu_[y[i]])
        sigma /= X.shape[0] - self.classes_.shape[0]
        self.cov_ = sigma
        self._cov_inv = inv(self.cov_)
        self.pi_ = df.groupby(by=0).size().to_numpy() / df.shape[0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.likelihood(X).argmax(axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        m = X.shape[0]
        d = X.shape[1]
        counts = self.pi_ * m
        consts = -d / 2 * np.log(2 * np.pi) - 1 / 2 * np.linalg.slogdet(self.cov_)[1] * np.linalg.slogdet(self.cov_)[0]
        likelihoods = np.zeros([m, self.classes_.shape[0]])
        for sample_index, sample in enumerate(X):
            sample_k_likelihoods = np.zeros(self.classes_.shape[0])
            for index, k in enumerate(self.classes_):
                log_pi_k = np.log(self.pi_[index])
                sample_calc = np.transpose(sample - self.mu_[index]) @ self._cov_inv @ (sample - self.mu_[index])
                sample_k_likelihoods[index] = log_pi_k + -0.5 * sample_calc + consts
            likelihoods[sample_index] = sample_k_likelihoods
        return likelihoods

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
