import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def decision_surface_helper(model: AdaBoost,
                            t, lims: np.array):
    return decision_surface(lambda x: model.partial_predict(x, t), lims[0],
                            lims[1], showscale=False)


def graph_for_surface(x: np.array, y: np.array, symbols: np.array, D=None):
    if D is None:
        return go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y.astype(int), symbol=symbols[y.astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))
    else:
        return go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=y.astype(int), symbol=symbols[y.astype(int)],
                                      colorscale=[custom[0], custom[-1]],
                                      line=dict(color="black", width=1), size=D))


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case

    test_loss = []
    train_loss = []
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    for i in range(1, n_learners + 1):
        test_loss.append(adaboost.partial_loss(test_X, test_y, i))
        train_loss.append(adaboost.partial_loss(train_X, train_y, i))
    train = go.Scatter(x=[x for x in range(1, 251)], y=train_loss, name="train error")
    test = go.Scatter(x=[x for x in range(1, 251)], y=test_loss, name="test error")
    go.Figure(data=[train, test], layout_title_text="Train and Test Loss Over Number of Iterations").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Decision Boundary Over {m} Iterations" for m in T])
    symbols = np.array([None, 'circle', 'x'])
    graph = graph_for_surface(test_X, test_y, symbols, None)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface_helper(adaboost, t, lims),
                        graph],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title="Decision Boundaries for Different Number of Iterations", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    test_best_loss = np.argmin(test_loss)
    accuracy = 1 - test_loss[test_best_loss]
    fig2 = go.Figure()
    fig2.add_traces([decision_surface_helper(adaboost, test_best_loss, lims), graph])
    fig2.update_layout(title=f"Best Size Loss is {test_best_loss + 1}, the Accuracy is {accuracy}", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.show()

    # Question 4: Decision surface with weighted samples
    D = adaboost.D_ / np.max(adaboost.D_) * 5
    fig3 = go.Figure()
    fig3.add_traces([decision_surface_helper(adaboost, adaboost.iterations_, lims),
                     graph_for_surface(test_X, test_y, symbols, D)])
    fig3.update_layout(title=f"Train Sample Proportional to Size", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig3.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
