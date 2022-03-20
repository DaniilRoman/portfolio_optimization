import pandas as pd
from pandas import DataFrame
from prophet import Prophet
from matplotlib import pyplot

from prophet.plot import plot_plotly, plot_components_plotly

# df = pd.read_csv('./data/data.csv')
# print(df.tail())

# df.plot()
# pyplot.show()

# m = Prophet()


def get_future_value(data: DataFrame, prophet: Prophet, predict_period: int = 30) -> float:
    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=predict_period)

    forecast = prophet.predict(future)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    return res["yhat"].iloc[0]


# get_future_value(df, m, 30)
