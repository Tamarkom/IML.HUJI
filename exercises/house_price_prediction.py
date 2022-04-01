import IMLearn
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    # clean bad data
    df = df[(df.price > 0) & (df.price != np.nan) & (df.bedrooms < 20) & (df.yr_built > 0) & ((df.bedrooms > 0) |
                                                                                              (df.bathrooms > 0))]

    # innovated \ new house
    df['is_renovated'] = 0
    df.loc[((df['yr_renovated'] >= 1990) | (df['yr_built'] >= 1990)), 'is_renovated'] = 1

    # dummies dor decades for yr_built
    df['yr_built'] = df['yr_built'] // 10 * 10
    df = pd.get_dummies(df, prefix='yr_built_', columns=['yr_built'])

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])

    df['living_area_ration_with_neighbours'] = (df['sqft_living'] / df['sqft_living15']) ** 0.5

    # only toilet and no shower
    df['no_shower'] = 0
    df.loc[(df['bathrooms'] < 1), 'no_shower'] = -1

    df['area_value'] = 0
    df.loc[((45.7 <= df['lat']) & (df['lat'] <= 47.71) & (-122.45 <= df['long']) & (df['long'] <= -121.92)), 'area_value'] = 1

    # drops
    response = df['price']
    df.drop(['id', 'yr_renovated', 'lat', 'long', 'price', 'date'], axis=1, inplace=True)
    return df, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        pearson_corr = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(x=X[feature], y=y, title="Pearson Correlation of {f}"
                                                  " and Response {corr}".format(f=feature, corr=pearson_corr))
        path = output_path + '\{feature}.png'.format(feature=feature)
        fig.write_image(path, format='png')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data('..\datasets\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y, '..\datasets\ex2_plots')

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = IMLearn.utils.split_train_test(x, y, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    results = []
    pos_std = []
    neg_std = []
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    linear_regressor = LinearRegression()
    x_axis = list(range(10, 101))
    for i in range(1, 101):
        cur_loss = []
        for _ in range(10):
            x_sample = train_x.sample(frac=i/100)
            y_sample = train_y.loc[x_sample.index]
            linear_regressor._fit(x_sample.to_numpy(), y_sample.to_numpy())
            cur_loss.append(linear_regressor._loss(test_x, test_y))
        loss = np.array(cur_loss)
        mean_loss = np.mean(loss)
        std_loss = np.std(loss)
        results.append(mean_loss)
        pos_std.append(mean_loss + 2 * std_loss)
        neg_std.append(mean_loss - 2 * std_loss)

    data = [go.Scatter(x=x_axis, y=results, name="Loss", mode='markers+lines', marker=dict(color='blue', opacity=0.7)),
           go.Scatter(x=x_axis, y=pos_std, fill=None, mode="lines", line=dict(color="lightgrey")),
           go.Scatter(x=x_axis, y=neg_std, fill='tonexty', mode="lines", line=dict(color="lightgrey"))]
    fig = go.Figure(data=data, layout=go.Layout(title="Loss Function of Sample Percentage",
                                                xaxis={"title": "Sample Percentage"},
                                                yaxis={"title": "Mean Square Loss"}))
    fig.show()
