import telepot
import configuration
from src.logic.data.data import StockData, ProfitabilityData

def notify(result: StockData):
    if not result.is_stock_growing or not result.profitability_data.is_profitable():
        return
    msg_to_send = __to_msg(result)
    first_photo_to_send = open(result.two_year_file_name, 'rb')
    second_photo_to_send = open(result.five_year_file_name, 'rb')

    bot = telepot.Bot(configuration.TELEGRAM_TOKEN)
    bot.getMe()
    bot.sendMediaGroup(
        chat_id=configuration.TELEGRAM_TO,
        media=[
            {'media' : first_photo_to_send, 'type' : 'photo', 'caption': msg_to_send, 'parse_mode': 'Markdown'},
            {'media' : second_photo_to_send, 'type' : 'photo'}
        ]
    )

def __profitability_data(profitability_data: ProfitabilityData) -> str:
    res = "\n"
    if profitability_data.ebitda_margins < 0:
        res += f"Ebitda margins: {profitability_data.ebitda_margins}\n"
    if profitability_data.operating_margins < 0:
        res += f"Operating margins: {profitability_data.operating_margins}\n"
    if profitability_data.forward_eps < 0:
        res += f"Forward EpS: {profitability_data.forward_eps}\n"
    if profitability_data.trailing_eps < 0:
        res += f"Trailing EpS: {profitability_data.ebitda_margins}\n"
    if profitability_data.netIncome_to_common < 0:
        res += f"Net income to common: {profitability_data.netIncome_to_common}\n"
    if res == "\n":
        return ""
    else:
        return res[:-2]

def __to_msg(result: StockData) -> str:
    stock_name_md_link = f'[{result.ticker_symbol}](https://finance.yahoo.com/quote/{result.ticker_symbol})'
    profitability = __profitability_data(result.profitability_data)
    industry = f'__{result.industry}__'
    return f'{stock_name_md_link} from {result.current_price} to {result.predict_price} ({result.currency})\n{result.stock_name}\n\n{industry}{profitability}'
