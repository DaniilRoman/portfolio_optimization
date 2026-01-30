import logging

from src.logic import stock_finder
from src.logic.data import data
from src.adapter.out.notify import notifier
from stock_names import etf_list

if __name__ == '__main__':
    notifier.send_text_message("=================")
    
    for etf in etf_list:
        try:
            stock_finder.run(etf)
        except data.SkipException as e:
            logging.error(e)
