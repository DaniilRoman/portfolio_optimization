from adapter.out.download import stock_data_downloader



def run(stock_name: str):
    print(f"Job `{stock_name}` created")

    # 1 - download prices
    prices = stock_data_downloader.construct_price_data(src, self.predict_days, self.repo, self.is_backtest)
    # 2 - make predictions
    # 3 - calculate optimal
    # 4 - notify user

