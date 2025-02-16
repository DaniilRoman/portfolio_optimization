import telepot
import re
import numpy
import configuration
from src.logic.data.data import StockData, ProfitabilityData

def notify(result: StockData):
    if not result.is_stock_growing or not result.profitability_data.is_profitable():
        print(f"Stock {result.stock_name} is not growing - will be skipped")
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

def __escape_markdown(text: str) -> str:
    """
    Helper function to escape special characters in Markdown.
    """
    escape_chars = r'\*_`['
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def __sector_allocation_pretty_print(d: dict) -> str:
    if not d:
        return ""
    
    # Sort the dictionary by value in descending order
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)
    
    pretty_str = ""
    for key, value in sorted_items:
        pretty_str += f"{key.capitalize()}: {value:.2%}\n"
    
    return f'\n\nSector Allocation:\n{pretty_str.strip()}\n'

def __format_top_holdings(top_holdings: numpy.ndarray) -> str:
    if top_holdings.size == 0:
        return ""
    
    # Sort the top_holdings array by the second value (weight)
    sorted_top_holdings = top_holdings[top_holdings[:, 1].argsort()[::-1]]

    holdings_str = ""
    for i, holding in enumerate(sorted_top_holdings, start=1):
        name = holding[0]
        weight = holding[1]
        holdings_str += f"{name} - {weight:.3f}\n"
    
    return f'\nTop Holdings:\n{holdings_str.strip()}'

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
    industry = f'__{result.industry}__' if result.industry else ''
    
    # New fields with conditional checks
    beta = f'Beta: {result.beta}\n' if result.beta else ''
    standard_deviation = f'Standard Deviation: {result.standard_deviation}\n' if result.standard_deviation else ''
    dividend_yield = f'Dividend Yield: {result.dividend_yield}\n' if result.dividend_yield else ''
    average_daily_volume = f'Average Volume: {result.average_daily_volume}\n' if result.average_daily_volume else ''
    assets_under_management = f'Assets Under Management: {result.assets_under_management}\n' if result.assets_under_management else ''
    expense_ratio = f'Expense Ratio: {result.expense_ratio}\n' if result.expense_ratio else ''
    description = f'Description: {result.description}\n' if result.description else ''
    
    return (
        f'{stock_name_md_link} from {result.current_price} to {result.predict_price} ({result.currency})\n'
        f'{result.stock_name}\n\n'
        f'{average_daily_volume}'
        f'{description}'
        f'{__escape_markdown(__format_top_holdings(result.top_holdings))}'
        f'{__escape_markdown(__sector_allocation_pretty_print(result.sector_allocation))}'
        f'{industry}\n'
        f'{profitability}'
        f'{beta}'
        f'{standard_deviation}'
        f'{dividend_yield}'
        f'{assets_under_management}'
        f'{expense_ratio}'
    ).strip()
