"""Sends analysis results to a Telegram chat via telepot; handles text messages and photo albums with chunking."""
import re

import numpy as np
import telepot

from config.configuration import settings
from src.logic.data.data import OptimizationResult, Portfolio, ProfitabilityData, StockData


def send_text_message(text: str) -> None:
    bot = telepot.Bot(settings.TELEGRAM_TOKEN)
    bot.getMe()
    bot.sendMessage(chat_id=settings.TELEGRAM_TO, text=text)


def notify(result: StockData) -> None:
    if not __is_notifyable(result):
        print(f"Stock {result.stock_name} is not growing - will be skipped")
        return
    msg_to_send = __to_msg(result)
    first_photo_to_send = open(result.two_year_file_name, "rb")
    second_photo_to_send = open(result.five_year_file_name, "rb")

    bot = telepot.Bot(settings.TELEGRAM_TOKEN)
    bot.getMe()
    bot.sendMediaGroup(
        chat_id=settings.TELEGRAM_TO,
        media=[
            {"media": first_photo_to_send, "type": "photo", "caption": msg_to_send, "parse_mode": "Markdown"},
            {"media": second_photo_to_send, "type": "photo"},
        ],
    )


def __is_notifyable(result: StockData) -> bool:
    return True  # TODO: add real filter once criteria are stable


def __escape_markdown(text: str) -> str:
    escape_chars = r"\*_`["
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def __sector_allocation_pretty_print(d: dict[str, float]) -> str:
    if not d:
        return ""

    epsilon = 1e-9
    filtered_items: list[tuple[str, float]] = []
    for key, value in d.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if val > epsilon:
            filtered_items.append((key, val))

    if not filtered_items:
        return ""

    sorted_items = sorted(filtered_items, key=lambda item: item[1], reverse=True)

    pretty_str = ""
    for key, value in sorted_items:
        pretty_str += f"{key.capitalize()}: {value:.2%}\n"

    return f"\n\nSector Allocation:\n{pretty_str.strip()}\n"


def __format_top_holdings(top_holdings: np.ndarray) -> str:
    if top_holdings.size == 0:
        return ""

    sorted_top_holdings = top_holdings[top_holdings[:, 1].astype(float).argsort()[::-1]]

    holdings_str = ""
    for holding in sorted_top_holdings:
        name = holding[0]
        weight = float(holding[1])
        holdings_str += f"{name} - {weight:.3f}\n"

    return f"\nTop Holdings:\n{holdings_str.strip()}"


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
    return res[:-2]


def __to_msg(result: StockData) -> str:
    stock_name_md_link = f"[{result.ticker_symbol}](https://finance.yahoo.com/quote/{result.ticker_symbol})"
    profitability = __profitability_data(result.profitability_data)
    industry = f"__{result.industry}__" if result.industry else ""

    beta = f"Beta: {result.beta}\n" if result.beta else ""
    standard_deviation = f"Standard Deviation: {result.standard_deviation}\n" if result.standard_deviation else ""
    dividend_yield = f"Dividend Yield: {result.dividend_yield}\n" if result.dividend_yield else ""
    average_daily_volume = f"Average Volume: {result.average_daily_volume}\n" if result.average_daily_volume else ""
    assets_under_management = f"Assets Under Management: {result.assets_under_management}\n" if result.assets_under_management else ""
    expense_ratio = f"Expense Ratio: {result.expense_ratio}\n" if result.expense_ratio else ""
    description = f"Description: {result.description}\n" if result.description else ""

    return (
        f"{stock_name_md_link} from {result.current_price} to {result.predict_price} ({result.currency})\n"
        f"{result.stock_name}\n\n"
        f"{average_daily_volume}"
        f"{description}"
        f"{__escape_markdown(__format_top_holdings(result.top_holdings))}"
        f"{__escape_markdown(__sector_allocation_pretty_print(result.sector_allocation))}"
        f"{industry}\n"
        f"{profitability}"
        f"{beta}"
        f"{standard_deviation}"
        f"{dividend_yield}"
        f"{assets_under_management}"
        f"{expense_ratio}"
    ).strip()


def send_optimization_result(result: OptimizationResult) -> None:
    message = __format_optimization_result(result)
    send_text_message(message)


def __format_portfolio(portfolio: Portfolio) -> str:
    if not portfolio.allocations:
        return "No ETFs to buy with current budget and constraints."

    lines: list[str] = []
    for alloc in portfolio.allocations:
        net_pct = (alloc.net_profit / alloc.total_cost * 100) if alloc.total_cost > 0 else 0
        forecast_vol = float(getattr(alloc.asset, "forecast_volatility", 0.0))
        lines.append(f"• *{alloc.asset.ticker_symbol}*: {alloc.quantity} shares")
        lines.append(f"  {alloc.asset.stock_name}")
        lines.append(f"  Cost: €{alloc.total_cost:.2f}")
        lines.append(f"  Net Profit: €{alloc.net_profit:.2f} ({net_pct:.1f}%)")
        lines.append(f"    - Capital Gain: €{alloc.capital_gain:.2f}")
        lines.append(f"    - Dividend Income: €{alloc.dividend_income:.2f}")
        lines.append(f"    - Expenses ({alloc.expense_ratio * 100:.2f}%): €{alloc.expense_fee:.2f}")
        if forecast_vol > 0:
            lines.append(f"    - Forecast Volatility: {forecast_vol * 100:.2f}%")
        lines.append("")

    total_cost = sum(a.total_cost for a in portfolio.allocations)
    total_profit = sum(a.net_profit for a in portfolio.allocations)
    total_gains = sum(a.capital_gain for a in portfolio.allocations)
    total_div = sum(a.dividend_income for a in portfolio.allocations)
    total_exp = sum(a.expense_fee for a in portfolio.allocations)

    lines.append(f"💰 *Total Investment:* €{total_cost:.2f}")
    lines.append(f"📈 *Total Net Profit:* €{total_profit:.2f}")
    lines.append(f"   - Capital Gains: €{total_gains:.2f}")
    lines.append(f"   - Dividend Income: €{total_div:.2f}")
    lines.append(f"   - Total Expenses: €{total_exp:.2f}")
    lines.append(f"   - Total Gross Profit: €{total_gains + total_div:.2f}")
    lines.append("")
    rm = portfolio.risk_metrics
    lines.append("⚠️ *Risk Metrics:*")
    lines.append(f"   - Volatility: {rm.volatility * 100:.1f}%")
    lines.append(f"   - Sector Concentration: {rm.sector_concentration:.3f}")
    lines.append(f"   - Company Overlap: {rm.company_overlap:.3f}")

    return "\n".join(lines)


def __format_optimization_result(result: OptimizationResult) -> str:
    return "\n".join([
        "📊 *Portfolio Optimization Results*",
        "",
        "⚠️ **Risk-Aware Optimization** (Minimizes risk while maximizing profit)",
        "",
        __format_portfolio(result.risk_aware),
        "",
        "📈 **Profit-Only Optimization** (Maximizes profit only)",
        "",
        __format_portfolio(result.profit_only),
    ])
