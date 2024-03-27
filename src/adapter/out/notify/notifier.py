import telepot
import configuration
from src.logic.data.data import StockData

def notify(result: StockData):
    msg_to_send = __to_msg(result)
    photo_to_send = open(result.file_name, 'rb')

    bot = telepot.Bot(configuration.TELEGRAM_TOKEN)
    bot.getMe()
    bot.sendPhoto(configuration.TELEGRAM_TO, caption=msg_to_send, parse_mode='Markdown', photo=photo_to_send)


def __to_msg(result: StockData) -> str:
    stock_name_md_link = f'[{result.ticker_symbol}](https://finance.yahoo.com/quote/{result.ticker_symbol})'
    common_msg = f'{stock_name_md_link} from {result.current_price} to {result.predict_price}\n{result.stock_name}'
    if result.predict_price <= result.current_price:
        return f'--- Filtered out {common_msg}'
    else: 
        return common_msg