import logging

from config.stock_names import etf_list
from src.adapter.out.notify import notifier
from src.adapter.out.optimization import optimizer_dispatcher as optimizer
from src.logic import stock_finder
from src.logic.data import data

if __name__ == '__main__':
    notifier.send_text_message("=================")
    
    analyses_results = []
    for etf in etf_list:
        try:
            analyses_result = stock_finder.run(etf)
            analyses_results.append(analyses_result)
        except data.SkipException as e:
            logging.error(e)
    optimization_result = optimizer.optimize(analyses_results)
    notifier.send_text_message(optimization_result)