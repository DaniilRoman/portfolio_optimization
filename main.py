import logging

from config.configuration import settings
from config.stock_names import resolve_universe
from src.adapter.out.notify import notifier
from src.adapter.out.optimization import optimizer_dispatcher as optimizer
from src.application import stock_finder
from src.domain.data import data

if __name__ == "__main__":
    notifier.send_text_message("=================")

    analyses_results = []
    for etf in resolve_universe(settings.UNIVERSE):
        try:
            analyses_result = stock_finder.run(etf)
            analyses_results.append(analyses_result)
        except data.SkipException as e:
            logging.error(e)
    optimization_result = optimizer.optimize(analyses_results)
    notifier.send_optimization_result(optimization_result)
