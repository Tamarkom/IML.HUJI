from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_diabetes

from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree
    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate
    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    eps = np.random.normal(scale=noise, size=n_samples)
    uniform_x = np.linspace(-1.2, 2, n_samples)
    df = pd.DataFrame(np.column_stack((uniform_x, f(uniform_x))), columns=['x', 'y'])
    noise_data = df.copy()
    noise_data['y'] += eps
    train_x, train_y, test_x, test_y = split_train_test(noise_data['x'], noise_data['y'], 2/3)
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=df['x'], y=df['y'],
                               name="Noiseless",
                               mode="lines+markers",
                               marker=dict(color="Black")),
                    go.Scatter(x=test_x, y=test_y,
                               name="Test",
                               mode="markers",
                               marker=dict(color="blueviolet")),
                    go.Scatter(x=train_x, y=train_y,
                               name="Train",
                               mode="markers",
                               marker=dict(color="paleturquoise"))])
    fig.update_layout(title="Split and noisy Data and Origin Data", showlegend=True).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_score = []
    validation_score = []
    best_error = np.inf
    best_degree = 0
    for k in range(11):
        train_error, test_error = cross_validate(PolynomialFitting(k), train_x.to_numpy(), train_y.to_numpy(),
                                                 mean_square_error)
        train_score.append(train_error)
        validation_score.append(test_error)
        if test_error < best_error:
            best_degree = k
            best_error = test_error
    train_df = pd.DataFrame(train_score, columns=['error'])
    train_df['kind'] = 'train'
    validation_df = pd.DataFrame(validation_score, columns=['error'])
    validation_df['kind'] = 'validation'
    df = pd.concat([train_df, validation_df])
    px.line(df, y='error', color='kind', title="Error Score of Validation and Train as function of Polynomial Degree"
                                               " with noise {} and number of samples {}"
            .format(noise, n_samples)).show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_polynomial = PolynomialFitting(best_degree)
    best_polynomial.fit(train_x.to_numpy(), train_y.to_numpy())
    rounded_error = round(mean_square_error(test_y.to_numpy(), best_polynomial.predict(test_x.to_numpy())), 2)
    print("The polynomial degree best fitted is {} and it test_error is {}".format(best_degree, rounded_error))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions
    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate
    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    diabetes_x, diabetes_y = load_diabetes(return_X_y=True)
    train_x = diabetes_x[:n_samples, :]
    train_y = diabetes_y[:n_samples]
    test_x = diabetes_x[n_samples:, :]
    test_y = diabetes_y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    train_ridge, validation_ridge, train_lasso, validation_lasso = [], [], [], []
    uniform_lambda = np.linspace(0.007, 3, n_evaluations)
    for k in uniform_lambda:
        ridge_train_error, ridge_validation_error = cross_validate(RidgeRegression(k), train_x,
                                                                   train_y, mean_square_error)
        lasso_train_error, lasso_validation_error = cross_validate(Lasso(alpha=k), train_x,
                                                                   train_y, mean_square_error)
        train_ridge.append(ridge_train_error)
        train_lasso.append(lasso_train_error)
        validation_ridge.append(ridge_validation_error)
        validation_lasso.append(lasso_validation_error)

    fig = go.Figure()
    fig.add_traces([go.Scatter(x=uniform_lambda, y=train_ridge,
                               name="Train Ridge",
                               mode="lines",
                               marker=dict(color="palevioletred")),
                    go.Scatter(x=uniform_lambda, y=validation_ridge,
                               name="Validation Ridge",
                               mode="lines",
                               marker=dict(color="peachpuff")),
                    go.Scatter(x=uniform_lambda, y=train_lasso,
                               name="Train Lasso",
                               mode="lines",
                               marker=dict(color="indigo")),
                    go.Scatter(x=uniform_lambda, y=validation_lasso,
                               name="Validation Lasso",
                               mode="lines",
                               marker=dict(color="firebrick")),
                    ])
    fig.update_layout(title="Lasso and Ridge Error using cross validation", showlegend=True).show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambda_lasso = validation_lasso.index(min(validation_lasso))
    best_lambda_lasso = uniform_lambda[best_lambda_lasso]
    best_lambda_ridge = validation_ridge.index(min(validation_ridge))
    best_lambda_ridge = uniform_lambda[best_lambda_ridge]
    lasso = Lasso(alpha=best_lambda_lasso)
    lasso.fit(train_x, train_y)
    print("The Error over lambda {} on Lasso is {}".format(best_lambda_lasso, mean_square_error(test_y,
                                                                                                lasso.predict(test_x))))
    ridge = RidgeRegression(lam=best_lambda_ridge)
    ridge.fit(train_x, train_y)
    print("The Error over lambda {} on Ridge is {}".format(best_lambda_ridge, ridge.loss(test_x, test_y)))

    least_square_model = LinearRegression()
    least_square_model.fit(train_x, train_y)
    print("The Error in Least Square Model is {}".format(least_square_model.loss(test_x, test_y)))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
