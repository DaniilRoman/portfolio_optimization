from pandas import DataFrame
from prophet import Prophet

from utils import round_precise


def predict_value(data: DataFrame, predict_period: int = 30) -> float:
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=predict_period)

    forecast = prophet.predict(future)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    res = res["yhat"].iloc[0]
    res = round(res, round_precise)
    return res
