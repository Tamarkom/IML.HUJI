import pandas as pd
import plotly.express
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly

pio.templates.default = "simple_white"
Q1_FORMAT = "({E}, {Var})"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    x = np.random.normal(10, 1, 1000)
    estimator = UnivariateGaussian()
    estimator = estimator.fit(x)
    print(Q1_FORMAT.format(E=estimator.mu_, Var=estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    diffs_list = []
    sample_sizes = []

    # take i first samples from 10 to 1000
    for i in range(10, 1001, 10):
        # calculate the diffs
        diffs_list.append(abs(10 - estimator.fit(x[:i]).mu_))
        sample_sizes.append(i)

    # plot the diffs
    df = pd.DataFrame({'sample size': sample_sizes, 'diffs': diffs_list})
    plotly.express.bar(df, x='sample size', y='diffs',
                       title='Distance Between True Value and Estimated Values of Expectations').show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = estimator.pdf(x)
    plotly.express.scatter(pdf, x=x, y=pdf, title='Probability Density Function',
                           labels={'x': 'samples', 'y': 'Density'}).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    x = np.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian()
    estimator = estimator.fit(x)
    print(estimator.mu_, '\n', estimator.cov_)

    # Question 5 - Likelihood evaluation
    func = np.linspace(-10, 10, 200)
    heatmap_list = []
    max_val = np.NINF
    for f1 in func:
        inner_list = []
        for f3 in func:
            value = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), sigma, x)
            # find the maximum log-likelihood for Q6
            if value > max_val:
                max_val, best_x, best_y = value, f1, f3
            inner_list.append(value)
        heatmap_list.append(inner_list[:])
    # TODO: add descriptions for axis
    go.Figure(data=go.Heatmap(x=func, y=func, z=heatmap_list), layout=go.Layout(title='Heatmap of log-likelihoods '
                                                                                      'Between -10 to 10')).show()

    # Question 6 - Maximum likelihood
    print("The Max log-likelihood is mu: [{},0,{},0], and the value is {}".format(best_x, best_y, max_val))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
