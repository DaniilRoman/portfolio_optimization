from pandas import DataFrame
from prophet import Prophet


def predict(data: DataFrame, predict_period: int = 30) -> (Prophet, any):
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=predict_period)

    forecast = prophet.predict(future)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return prophet, res
