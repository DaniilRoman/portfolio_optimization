from pandas import DataFrame
from prophet import Prophet


def predict(data: DataFrame, predict_period: int = 30) -> (Prophet, any):
    data = data.reset_index(drop=True)
    prophet = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=predict_period)

    forecast = prophet.predict(future)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return prophet, res
