import telepot
import configuration
from src.logic.data.data import StockData, ProfitabilityData

def notify(result: StockData):
    if not result.is_stock_growing:
        return
    msg_to_send = __to_msg(result)
    photo_to_send = open(result.file_name, 'rb')

    bot = telepot.Bot(configuration.TELEGRAM_TOKEN)
    bot.getMe()
    bot.sendPhoto(configuration.TELEGRAM_TO, caption=msg_to_send, parse_mode='Markdown', photo=photo_to_send)

def __profitability_data(profitability_data: ProfitabilityData) -> str:
    res = "\n"
    if profitability_data.ebitda_margins < 0:
        res += f"ebitda_margins: {profitability_data.ebitda_margins}\n"
    if profitability_data.operating_margins < 0:
        res += f"operating_margins: {profitability_data.operating_margins}\n"
    if profitability_data.forward_eps < 0:
        res += f"forward_eps: {profitability_data.forward_eps}\n"
    if profitability_data.trailing_eps < 0:
        res += f"trailing_eps: {profitability_data.ebitda_margins}\n"
    if profitability_data.netIncome_to_common < 0:
        res += f"netIncome_to_common: {profitability_data.netIncome_to_common}\n"
    if res == "\n":
        return ""
    else:
        return res

def __to_msg(result: StockData) -> str:
    stock_name_md_link = f'[{result.ticker_symbol}](https://finance.yahoo.com/quote/{result.ticker_symbol})'
    profitability = __profitability_data(result.profitability_data)
    return f'{stock_name_md_link} from {result.current_price} to {result.predict_price} ({result.currency})\n{result.stock_name}\n{result.industry}{profitability}'