import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=['Date']).dropna()
    df = df[df['Temp'] > -70]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('..\datasets\City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = df[df['Country'] == 'Israel']
    px.scatter(israel_df, x="DayOfYear", y='Temp', color="Year",
               title="Temp Over The Years on Days of the years in Israel").show()
    month_temp_std = israel_df.groupby('Month')['Temp'].agg('std')
    px.bar(month_temp_std, x=range(1, 13), y='Temp', title="std of for Temp in every Month in Israel").show()

    # Question 3 - Exploring differences between countries
    country_month_temp_std = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp': ['std', 'mean']})
    country_month_temp_std.columns = ["_".join(a) for a in country_month_temp_std.columns.to_flat_index()]
    px.line(country_month_temp_std, color="Country_", error_y='Temp_std', y='Temp_mean', x='Month_',
            title="Average Temp by Country and Month").show()

    # Question 4 - Fitting model for different values of `k`
    israel_y = israel_df['Temp']
    israel_x = israel_df['DayOfYear']
    train_x, train_y, test_x, test_y = split_train_test(israel_x,
                                                        israel_y, 0.75)
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    test_x = test_x.to_numpy()
    loss = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_x, train_y)
        mse = round(poly_fit.loss(test_x, test_y), 2)
        loss.append(mse)
        print("Test Error for degree {k} is {mse}".format(k=k, mse=mse))
    px.bar(x=range(1, 11), y=loss, title="Mean Square Error for Degrees 1-10").show()

    # Question 5 - Evaluating fitted model on different countries
    poly_fit = PolynomialFitting(5)
    poly_fit.fit(train_x, train_y)

    country_list = []
    loss_list = []
    for country in df['Country'].unique():
        if country != 'Israel':
            country_df = df[df['Country'] == country]
            country_df_y = country_df['Temp']
            country_df_x = country_df['DayOfYear']
            country_list.append(country)
            loss_list.append(poly_fit.loss(country_df_x, country_df_y))
    px.bar(x=country_list, y=loss_list, title="MSE for Countries over Israel Model").show()