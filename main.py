import logging

from src.logic import stock_finder
from src.logic.data import data
from stock_names import etf_list

if __name__ == '__main__':
    for i in range(len(etf_list)):
        try:
            stock_finder.run()
        except data.SkipException as e:
            logging.error(e)