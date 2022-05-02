import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from typing import List

from tqdm import tqdm

from optimization_job_repo import OptimizationRepository
from services.analyze import construct_result, print_result
from data import StockOptimizationJob, OptimizationJobStatus, StockLimit, StockLimitType, PriceData
from services.stock_data_downloader import construct_price_data, download_current_price
from services.stock_limit_transformer import transform_stock_limit
from services.stock_optimize import optimize
from utils import get_prev_day, OnlyPutBlockingQueue


def get_sp500_stocks():
    # There are 2 tables on the Wikipedia page
    # we want the first table

    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    second_table = payload[1]

    df = first_table

    symbols = df['Symbol'].values.tolist()
    excluded_stocks = ["BRK.B", "BF.B", "MMM", "AES", "AFL", "A", "ABT", "ADBE"]
    return list(filter(lambda s: s not in excluded_stocks, symbols))


def create_optimization_task_step(stock_names: List[str], stock_limit: StockLimit, budget: int,
                                  predict_period_days: int, is_backtest: bool,
                                  repo: OptimizationRepository) -> StockOptimizationJob:
    job = StockOptimizationJob(stock_names, stock_limit, budget, predict_period_days, is_backtest)
    repo.save_job(job)
    return job


class PriceGetter(object):
    def __init__(self, predict_days, repo, is_backtest):
        self.tmp_dict = OnlyPutBlockingQueue()
        self.predict_days = predict_days
        self.repo = repo
        self.is_backtest = is_backtest

    def __call__(self, src):
        res = construct_price_data(src, self.predict_days, self.repo, self.is_backtest)
        self.tmp_dict.put(src, res)


def filter_negative_stock(stock_price: PriceData) -> bool:
    if stock_price.predict_price < stock_price.current_price:
        print(f"Filter: {stock_price}")
        return False
    else:
        return True


def download_and_predict_step(job: StockOptimizationJob, repo: OptimizationRepository, parallelism: int = 50):
    print("STEP: download and predict")
    job.status = OptimizationJobStatus.PREPARING_DATA
    repo.save_job(job)

    task_object = PriceGetter(job.predict_period_days, repo, job.is_backtest)
    # with ThreadPoolExecutor(parallelism) as executor:
    #     executor.map(task_object, tqdm(job.stock_names))

    with ThreadPoolExecutor(parallelism) as executor:
        stock_price_futures = [executor.submit(task_object, stock_name) for stock_name in tqdm(job.stock_names)]
        for future in as_completed(stock_price_futures):
            # url = future_to_url[future]
            future.result()

    prices = [task_object.tmp_dict.queue[i] for i in job.stock_names]
    prices = list(filter(filter_negative_stock, prices))

    job.current_prices = [prices_pair.current_price for prices_pair in prices]
    job.predicted_prices = [prices_pair.predict_price for prices_pair in prices]
    job.real_prices = [prices_pair.real_future_price for prices_pair in prices]

    filtered_stock_names = [prices_pair.stock_name for prices_pair in prices]

    def filter_max_stock_count(stock_name, stock_count):
        if stock_name in filtered_stock_names:
            return True
        else:
            return False

    job.max_stock_count_list = [y for x, y in
                                list(filter(filter_max_stock_count, zip(job.stock_names, job.max_stock_count_list)))]
    job.stock_names = filtered_stock_names


def optimization_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Optimization")
    job.status = OptimizationJobStatus.OPTIMIZATION
    repo.save_job(job)

    best_set = optimize(job)
    job.best_set = best_set
    job.status = OptimizationJobStatus.FINISHED
    repo.save_job(job)


def construct_result_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Construct result")
    opt_result = construct_result(job)
    repo.save_opt_result(opt_result)
    print_result(opt_result)

    if job.is_backtest:
        finish_sp500_price = download_current_price("SPY", repo)
        start_sp500_price = download_current_price("SPY", repo, get_prev_day(job.predict_period_days))
        print("S&P500 price: " + str(round(finish_sp500_price, 5)))
        print("S&P500 price change: " + str(round(finish_sp500_price - start_sp500_price, 5)))
        print(
            "S&P500 price change: " + str(
                round((finish_sp500_price - start_sp500_price) / start_sp500_price * 100, 5)) + "%")

    return opt_result


def stock_limit_transform_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Stock limit transformation")
    transform_stock_limit(job, repo)
    repo.save_job(job)


def run_etl(stock_names: List[str], stock_limit: StockLimit, budget: int, predict_period_days: int,
            repo: OptimizationRepository, parallelism: int, is_backtest: bool):
    job = create_optimization_task_step(stock_names, stock_limit, budget, predict_period_days, is_backtest, repo)
    print(f"Job `{job._id}` created")
    download_and_predict_step(job, repo, parallelism)
    stock_limit_transform_step(job, repo)
    optimization_step(job, repo)
    construct_result_step(job, repo)


if __name__ == '__main__':
    start = time.time()
    stock_names = ["DIDI", "MSFT", "AMD", "BABA", "NVDA", "AAL"]
    # stock_names = ['AOS', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADM', 'ADP', 'AAP', 'AIG', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AMD', 'AEE', 'AAL', 'AEP', 'AXP', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ANET', 'AIZ', 'T', 'ATO', 'ADSK', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BBWI', 'BAX', 'BDX', 'WRB', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CDAY', 'CERN', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISH', 'DIS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FAST', 'FRT', 'FDX', 'FITB', 'FRC', 'FE', 'FIS', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'AJG', 'GRMN', 'IT', 'GE', 'GNRC', 'GD', 'GIS', 'GPC', 'GILD', 'GL', 'GPN', 'GM', 'GS', 'GWW', 'HAL', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HII', 'HBAN', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'FB', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OGN', 'OTIS', 'PCAR', 'PKG', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PENN', 'PNR', 'PEP', 'PKI', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SBNY', 'SPG', 'SWKS', 'SJM', 'SNA', 'SEDG', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TWTR', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'V', 'VNO', 'VMC', 'WAB', 'WMT', 'WBA', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WRK', 'WY', 'WHR', 'WMB', 'WTW', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']
    # stock_names = get_sp500_stocks()
    # stock_names = list(filter(lambda x: x not in ["BRK.B", "BF.B", "MMM", "AES", "AFL", "A", "ABT", "ADBE", "ARE", "AKAM", "ALK", "ALB", "ALGN", "ALLE", "LNT", "ALL", "GOOGL"], stock_names[10:30]))
    stock_limit = StockLimit(StockLimitType.PERCENT, common_limit=30)
    BUDGET = 2000
    PREDICT_PERIOD_DAYS = 30

    repo = OptimizationRepository()
    run_etl(stock_names, stock_limit, BUDGET, PREDICT_PERIOD_DAYS, repo, parallelism=100, is_backtest=False)
    print(f"Time: {time.time() - start}")
