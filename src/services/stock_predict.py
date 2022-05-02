from pandas import DataFrame
from prophet import Prophet

from optimization_job_repo import OptimizationRepository
from utils import get_next_day


def predict_value(data: DataFrame, predict_period: int = 30) -> float:
    prophet = Prophet()
    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=predict_period)

    forecast = prophet.predict(future)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    res = res["yhat"].iloc[0]
    res = round(res, 2)
    return res


def get_predict_value(stock_name: str, data: DataFrame, repo: OptimizationRepository,
                      predict_period: int = 30) -> float:
    exist_price = repo.get_saved_stock_price(get_next_day(predict_period), stock_name)
    if exist_price is not None:
        return exist_price

    res = predict_value(data, predict_period)

    repo.save_stock_predicted_price(get_next_day(predict_period), stock_name, res)
    print(f"Predicted: {stock_name}")
    return res
