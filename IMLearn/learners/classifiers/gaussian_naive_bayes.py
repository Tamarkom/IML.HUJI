from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import slogdet, inv
import pandas as pd

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        df = pd.concat([pd.DataFrame(y), pd.DataFrame(X)], axis=1, ignore_index=True)
        self.pi_ = df.groupby(by=0).size().to_numpy() / df.shape[0]
        self.mu_ = df.groupby(by=0).mean().to_numpy()
        num_classes = self.classes_.shape[0]
        num_features = X.shape[1]
        sigma = np.zeros([num_classes, num_features])
        for i in range(X.shape[0]):
            k_index = np.where(self.classes_ == y[i])[0][0]
            sigma[k_index] += np.diag(np.outer(X[i] - self.mu_[k_index], (X[i] - self.mu_[k_index])))
        for class_number, k in enumerate(self.classes_):
            sigma[class_number] /= counts[class_number] - num_classes
        self.vars_ = sigma

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
        consts = -d / 2 * np.log(2 * np.pi)
        likelihoods = np.zeros([m, self.classes_.shape[0]])
        for sample_index, sample in enumerate(X):
            sample_k_likelihoods = np.zeros(self.classes_.shape[0])
            for index, k in enumerate(self.classes_):
                log_pi_k = np.log(self.pi_[index])
                log_det = -0.5 * np.log(np.prod(self.vars_[index]))
                sample_calc = np.transpose(sample - self.mu_[index]) @ np.diag(1 / self.vars_[k]) @ (sample - self.mu_[index])
                sample_k_likelihoods[index] = consts - 1/2 * sample_calc + log_pi_k + log_det
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
