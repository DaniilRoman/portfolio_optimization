from flask import Flask, request
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

from data import StockLimit
from etl import run_etl_async
from services.stock_data_downloader import search_stocks, download_current_price
from optimization_job_repo import OptimizationRepository
from stock_names import russian_stocks

repo = OptimizationRepository()
executor = ThreadPoolExecutor(10)
app = Flask(__name__)
CORS(app)


@app.route("/stock/<stock_name>/price")
def get_price_endpoint(stock_name):
    return {"price": download_current_price(stock_name, repo)}


@app.route("/stock/<stock_name_query>/search")
def search_stocks_endpoint(stock_name_query):
    return {"stocks": search_stocks(stock_name_query)}


@app.route("/stock/rus")
def rus_stocks_endpoint():
    return {"stocks": russian_stocks}


@app.route("/job/<task_id>/status")
def get_status_endpoint(task_id):
    job = repo.get_job(task_id)
    return {"status": job.status.value}


@app.route("/job/<task_id>/result")
def get_result_endpoint(task_id):
    result = repo.get_opt_result(task_id)
    return {"result": result}


@app.route("/job/run", methods=['POST'])
def start_optimization_endpoint():
    data = request.json
    print(data)
    job_id = run_etl_async(data['stock_names'], StockLimit.from_dict(data['stock_limit']), data['budget'],
                           data['predicted_period_days'],
                           repo, parallelism=50, is_backtest=False, executor=executor)
    return {"job_id": job_id}


if __name__ == '__main__':
    app.run(port=9010)
