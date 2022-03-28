from pandas import DataFrame
from prophet import Prophet


def get_future_value(data: DataFrame, predict_period: int = 30) -> float:
    prophet = Prophet()
    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=predict_period)

    forecast = prophet.predict(future)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    return res["yhat"].iloc[0]
