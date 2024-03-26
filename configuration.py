from decouple import config
import logging

TELEGRAM_TO = config("TELEGRAM_TO")
TELEGRAM_TOKEN = config("TELEGRAM_TOKEN")
GET_AND_INCREMENT_COUNTER_URL = config("GET_AND_INCREMENT_COUNTER_URL")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')