import telepot
import configuration
from src.logic.data.data import StockData

def notify(result: StockData):
    msg_to_send = __to_msg(result)
    photo_to_send = open(result.file_name, 'rb')

    bot = telepot.Bot(configuration.TELEGRAM_TOKEN)
    bot.getMe()
    bot.sendMessage(configuration.TELEGRAM_TO, msg_to_send, parse_mode='Markdown')
    bot.sendPhoto(configuration.TELEGRAM_TO, photo=photo_to_send)


def __to_msg(result: StockData) -> str:
    return f'{result.stock_name} from {result.current_price} to {result.predict_price}'