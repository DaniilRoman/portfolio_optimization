import logging

from src.logic import stock_finder
from src.logic.data import data

if __name__ == '__main__':
    try:
        stock_finder.run()
    except data.SkipException as e:
        logging.error(e)