from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Mapping
from datetime import date, timedelta
from html import escape
from pathlib import Path
from urllib.parse import urlencode, urlparse

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


WATCHLIST_DB_PATH = Path(__file__).with_name(".watchlist.db")
TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9^.=-]{0,19}$")
INDICATOR_LOOKBACK_DAYS = 400
FUNDAMENTALS_CACHE_SECONDS = 900
YAHOO_NETWORK_RETRIES = 2
yf.config.network.retries = YAHOO_NETWORK_RETRIES
MOMENTUM_COLOURS = {
    "positive": "#22c55e",
    "negative": "#ef6461",
    "neutral": "#8b919d",
}
METRIC_GRID_STYLES = """
.metric-grid {
    display: grid;
    gap: 0.25rem 1.5rem;
    margin-bottom: 1.5rem;
    width: 100%;
}
.metric-grid--latest {
    grid-template-columns: repeat(4, minmax(0, 1fr));
}
.metric-grid--fundamentals {
    grid-template-columns: repeat(2, minmax(0, 1fr));
}
.market-metric {
    min-width: 0;
    padding: 0.75rem 0;
}
.metric-label,
.metric-value {
    font-size: 1rem !important;
    line-height: 1.4;
    margin: 0;
}
.metric-label {
    color: #a4a9b3;
    font-weight: 600;
}
.metric-value {
    font-weight: 600;
    margin-top: 0.25rem;
    overflow-wrap: anywhere;
}
.metric-value.positive { color: #22c55e; }
.metric-value.negative { color: #ef6461; }
.metric-value.neutral { color: inherit; }
@media (max-width: 900px) {
    .metric-grid--latest {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 520px) {
    .metric-grid--latest,
    .metric-grid--fundamentals {
        grid-template-columns: 1fr;
    }
}
"""
NEWS_STYLES = """
.news-list {
    list-style: none;
    margin: 0 0 1.5rem;
    padding: 0;
}
.news-item {
    border-bottom: 1px solid rgba(164, 169, 179, 0.20);
    padding: 0.875rem 0;
}
.news-item:last-child {
    border-bottom: none;
}
.news-link {
    color: inherit !important;
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.4;
    text-decoration: none !important;
}
.news-link:hover,
.news-link:focus-visible {
    text-decoration: underline !important;
}
.news-meta {
    color: #a4a9b3;
    font-size: 0.875rem;
    margin-top: 0.25rem;
}
"""

EXCHANGE_NAMES = {
    "ASE": "NYSE AMERICAN",
    "CMX": "COMEX",
    "DJI": "DOW JONES",
    "NASDAQGS": "NASDAQ",
    "NASDAQGM": "NASDAQ",
    "NASDAQCM": "NASDAQ",
    "NEW YORK COMMODITY EXCHANGE": "COMEX",
    "NCM": "NASDAQ",
    "NGM": "NASDAQ",
    "NMS": "NASDAQ",
    "NYM": "NYMEX",
    "NYQ": "NYSE",
    "PCX": "NYSE ARCA",
    "SNP": "S&P INDEX",
}

COMMON_MARKETS = {
    "AAPL": ("Apple Inc.", "NASDAQ"),
    "MSFT": ("Microsoft Corporation", "NASDAQ"),
    "NVDA": ("NVIDIA Corporation", "NASDAQ"),
    "TSLA": ("Tesla, Inc.", "NASDAQ"),
    "^GSPC": ("S&P 500", "S&P INDEX"),
    "^NDX": ("NASDAQ-100", "NASDAQ GIDS"),
    "^DJI": ("Dow Jones Industrial Average", "DOW JONES"),
    "GC=F": ("Gold Futures", "COMEX"),
    "CL=F": ("Crude Oil Futures", "NYMEX"),
    "SI=F": ("Silver Futures", "COMEX"),
}


def clean_ticker(ticker: str) -> str:
    """Remove extra spaces and use the uppercase ticker format."""
    return ticker.strip().upper()


def validate_ticker(ticker: str) -> str:
    """Return a normalized ticker or raise when its format is unsafe."""
    cleaned_ticker = clean_ticker(ticker)
    if not TICKER_PATTERN.fullmatch(cleaned_ticker):
        raise ValueError("Enter a valid ticker symbol.")
    return cleaned_ticker


def selected_ticker_from_query(query_params: Mapping[str, object]) -> str:
    """Return one validated ticker selected through the page URL."""
    selected_ticker = query_params.get("ticker", "")
    if isinstance(selected_ticker, list):
        selected_ticker = selected_ticker[-1] if selected_ticker else ""
    if not isinstance(selected_ticker, str):
        return ""

    try:
        return validate_ticker(selected_ticker)
    except ValueError:
        return ""


def clear_selected_ticker_query() -> None:
    """Clear a watchlist URL selection when the search field is edited."""
    if "ticker" in st.query_params:
        del st.query_params["ticker"]
    st.session_state.pop("watchlist_query_ticker", None)


def initialize_watchlist(database_path: Path = WATCHLIST_DB_PATH) -> None:
    """Create the local ordered watchlist table when it does not exist."""
    with sqlite3.connect(database_path, timeout=5) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                position INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL UNIQUE
            )
            """
        )


def load_watchlist(database_path: Path = WATCHLIST_DB_PATH) -> list[str]:
    """Load watchlist tickers in their original order of addition."""
    initialize_watchlist(database_path)
    with sqlite3.connect(database_path, timeout=5) as connection:
        rows = connection.execute(
            "SELECT ticker FROM watchlist ORDER BY position"
        ).fetchall()
    return [row[0] for row in rows]


def add_ticker_to_watchlist(
    ticker: str, database_path: Path = WATCHLIST_DB_PATH
) -> bool:
    """Add one validated ticker and return whether a new row was stored."""
    cleaned_ticker = validate_ticker(ticker)
    initialize_watchlist(database_path)
    with sqlite3.connect(database_path, timeout=5) as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)",
            (cleaned_ticker,),
        )
    return cursor.rowcount == 1


def remove_ticker_from_watchlist(
    ticker: str, database_path: Path = WATCHLIST_DB_PATH
) -> bool:
    """Remove one validated ticker and return whether a row was deleted."""
    cleaned_ticker = validate_ticker(ticker)
    initialize_watchlist(database_path)
    with sqlite3.connect(database_path, timeout=5) as connection:
        cursor = connection.execute(
            "DELETE FROM watchlist WHERE ticker = ?",
            (cleaned_ticker,),
        )
    return cursor.rowcount == 1


def latest_daily_change(closing_prices: pd.Series) -> tuple[float, float]:
    """Return the latest close and its change from the previous close."""
    available_prices = pd.to_numeric(closing_prices, errors="coerce").dropna()
    if len(available_prices) < 2:
        raise ValueError("At least two closing prices are required.")

    previous_price = float(available_prices.iloc[-2])
    current_price = float(available_prices.iloc[-1])
    if previous_price == 0:
        raise ValueError("The previous closing price cannot be zero.")

    percentage_change = ((current_price - previous_price) / previous_price) * 100
    return current_price, percentage_change


def readable_exchange_name(exchange: str | None) -> str:
    """Convert Yahoo's exchange code into a readable display name."""
    if not exchange:
        return "MARKET"

    cleaned_exchange = exchange.strip().upper()
    return EXCHANGE_NAMES.get(cleaned_exchange, cleaned_exchange)


def indicator_calculation_start(start_date: str) -> str:
    """Return an earlier date that supplies enough indicator history."""
    try:
        selected_start = pd.Timestamp(start_date)
    except (TypeError, ValueError) as error:
        raise ValueError("Enter a valid start date.") from error
    if pd.isna(selected_start):
        raise ValueError("Enter a valid start date.")
    return (
        selected_start - pd.Timedelta(days=INDICATOR_LOOKBACK_DAYS)
    ).date().isoformat()


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker: str, start_date: str) -> pd.DataFrame:
    """Download through Yahoo's newest row with extra indicator history."""
    data = yf.download(
        ticker,
        start=indicator_calculation_start(start_date),
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    # Raising prevents Streamlit from caching a temporary empty Yahoo response.
    if data.empty or "Close" not in data.columns:
        raise ValueError("Yahoo Finance returned no closing-price data.")

    data.index.name = "Date"
    return data


def add_technical_indicators(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Add close-based 50D/200D averages and Wilder-style 14D RSI."""
    if "Close" not in stock_data.columns:
        raise ValueError("Closing-price data is required.")

    enriched_data = stock_data.copy()
    closing_prices = pd.to_numeric(enriched_data["Close"], errors="coerce")
    enriched_data["MA_50"] = closing_prices.rolling(
        window=50,
        min_periods=50,
    ).mean()
    enriched_data["MA_200"] = closing_prices.rolling(
        window=200,
        min_periods=200,
    ).mean()

    close_changes = closing_prices.diff()
    gains = close_changes.clip(lower=0)
    losses = -close_changes.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50.0)
    rsi = rsi.mask((average_gain > 0) & (average_loss == 0), 100.0)
    enriched_data["RSI_14"] = rsi
    return enriched_data


def filter_data_from_start(
    stock_data: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    """Return rows on or after the user-selected start date."""
    try:
        selected_timestamp = pd.Timestamp(start_date)
    except (TypeError, ValueError) as error:
        raise ValueError("Enter a valid start date.") from error
    if pd.isna(selected_timestamp):
        raise ValueError("Enter a valid start date.")
    selected_start = selected_timestamp.date()

    available_dates = pd.to_datetime(stock_data.index, errors="coerce")
    visible_mask = [
        not pd.isna(value) and value.date() >= selected_start
        for value in available_dates
    ]
    visible_data = stock_data.loc[visible_mask].copy()
    if visible_data.empty:
        raise ValueError(
            "No trading data is available on or after the selected start date."
        )
    return visible_data


def latest_market_summary(stock_data: pd.DataFrame) -> dict[str, object]:
    """Return newest prices, volume, indicators, and one-day return."""
    if "Close" not in stock_data.columns:
        raise ValueError("Closing-price data is required.")

    closing_prices = pd.to_numeric(stock_data["Close"], errors="coerce")
    available_closes = closing_prices.dropna()
    if available_closes.empty:
        raise ValueError("At least one closing price is required.")

    latest_date = available_closes.index[-1]
    latest_row = stock_data.loc[latest_date]

    def available_number(column: str) -> float | None:
        if column not in stock_data.columns:
            return None
        try:
            value = float(latest_row[column])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    daily_return = None
    if len(available_closes) >= 2:
        previous_close = float(available_closes.iloc[-2])
        current_close = float(available_closes.iloc[-1])
        if previous_close != 0:
            daily_return = ((current_close / previous_close) - 1) * 100

    return {
        "date": pd.Timestamp(latest_date),
        "open": available_number("Open"),
        "high": available_number("High"),
        "low": available_number("Low"),
        "volume": available_number("Volume"),
        "ma_50": available_number("MA_50"),
        "ma_200": available_number("MA_200"),
        "rsi": available_number("RSI_14"),
        "return": daily_return,
    }


def momentum_direction(return_value: float | None) -> str:
    """Return the semantic colour direction for a daily return."""
    if return_value is None or not math.isfinite(return_value):
        return "neutral"
    if return_value > 0:
        return "positive"
    if return_value < 0:
        return "negative"
    return "neutral"


def _available_number(value: object) -> float | None:
    """Return a finite numeric value or None for unavailable metadata."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_compact_number(value: float, currency: bool = False) -> str:
    """Format large values with familiar thousand-to-trillion suffixes."""
    absolute_value = abs(value)
    for threshold, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if absolute_value >= threshold:
            formatted = f"{value / threshold:,.2f}{suffix}"
            return f"${formatted}" if currency else formatted
    formatted = f"{value:,.2f}"
    return f"${formatted}" if currency else formatted


def _format_market_date(value: object) -> str | None:
    """Return a readable date for Yahoo timestamps or date-like values."""
    try:
        if isinstance(value, (int, float)):
            timestamp = pd.to_datetime(value, unit="s", utc=True)
        else:
            timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def build_metric_grid(
    metrics: list[tuple[str, str, str]],
    layout: str,
    accessible_name: str,
) -> str:
    """Build a responsive, accessible metric grid with balanced type sizes."""
    if layout not in {"latest", "fundamentals"}:
        raise ValueError("Unknown metric-grid layout.")

    metric_rows = []
    for label, value, direction in metrics:
        safe_direction = (
            direction
            if direction in {"positive", "negative", "neutral"}
            else "neutral"
        )
        metric_rows.append(
            '<div class="market-metric" role="listitem">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value {safe_direction}">{escape(value)}</div>'
            "</div>"
        )

    return f"""
        <div class="metric-grid metric-grid--{layout}" role="list"
             aria-label="{escape(accessible_name)}">
            {''.join(metric_rows)}
        </div>
    """


def format_fundamental_metrics(
    market_info: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Format only the company fundamentals Yahoo currently provides."""
    metrics: list[tuple[str, str]] = []

    trailing_pe = _available_number(market_info.get("trailingPE"))
    if trailing_pe is not None:
        metrics.append(("P/E Ratio", f"{trailing_pe:,.2f}"))

    dividend_rate = _available_number(market_info.get("dividendRate"))
    if dividend_rate is not None:
        metrics.append(("Dividend", f"${dividend_rate:,.2f}"))

    trailing_eps = _available_number(market_info.get("trailingEps"))
    if trailing_eps is not None:
        metrics.append(("EPS", f"${trailing_eps:,.2f}"))

    ex_dividend_date = _format_market_date(
        market_info.get("exDividendDate")
    )
    if ex_dividend_date:
        metrics.append(("Ex-Dividend Date", ex_dividend_date))

    market_cap = _available_number(market_info.get("marketCap"))
    if market_cap is not None:
        metrics.append(
            ("Market Cap", _format_compact_number(market_cap, currency=True))
        )

    shares_outstanding = _available_number(
        market_info.get("sharesOutstanding")
    )
    if shares_outstanding is not None:
        metrics.append(
            (
                "Shares Outstanding",
                _format_compact_number(shares_outstanding),
            )
        )

    return metrics


def estimate_forward_dividend_rate(dividends: pd.Series) -> float | None:
    """Estimate an annual rate from the latest recurring dividend payment."""
    if not isinstance(dividends, pd.Series) or dividends.empty:
        return None

    available_dividends = pd.to_numeric(
        dividends,
        errors="coerce",
    ).dropna()
    available_dividends = available_dividends[available_dividends > 0]
    if available_dividends.empty:
        return None

    payment_frequency = None
    valid_dates = pd.Series(dtype="datetime64[ns, UTC]")
    if isinstance(available_dividends.index, pd.DatetimeIndex):
        payment_dates = pd.to_datetime(
            available_dividends.index,
            errors="coerce",
            utc=True,
        )
        valid_dates = pd.Series(payment_dates).dropna().sort_values()
    if len(valid_dates) >= 2:
        payment_gaps = valid_dates.diff().dt.days.dropna()
        if not payment_gaps.empty:
            median_gap = float(payment_gaps.tail(12).median())
            if median_gap <= 45:
                payment_frequency = 12
            elif median_gap <= 150:
                payment_frequency = 4
            elif median_gap <= 250:
                payment_frequency = 2
            else:
                payment_frequency = 1

    if payment_frequency is None:
        return float(available_dividends.tail(4).sum())
    return float(available_dividends.iloc[-1]) * payment_frequency


def collect_fundamental_info(ticker: str) -> dict[str, object]:
    """Collect fundamentals with independent fallbacks for Yahoo outages."""
    instrument = yf.Ticker(ticker)
    market_info: dict[str, object] = {}
    detailed_info_loaded = False

    try:
        detailed_info = instrument.get_info()
        if isinstance(detailed_info, Mapping):
            market_info.update(detailed_info)
            detailed_info_loaded = bool(detailed_info)
    except Exception:
        pass

    if not detailed_info_loaded:
        try:
            instrument = yf.Ticker(ticker)
            detailed_info = instrument.get_info()
            if isinstance(detailed_info, Mapping):
                market_info.update(detailed_info)
        except Exception:
            pass

    def numeric_value_missing(key: str) -> bool:
        return _available_number(market_info.get(key)) is None

    fast_info: dict[str, object] = {}
    if any(
        numeric_value_missing(key)
        for key in ("trailingPE", "marketCap", "sharesOutstanding")
    ):
        try:
            fast_info.update(dict(instrument.get_fast_info()))
        except Exception:
            pass

    if numeric_value_missing("marketCap"):
        market_info["marketCap"] = fast_info.get("marketCap")
    if numeric_value_missing("sharesOutstanding"):
        market_info["sharesOutstanding"] = fast_info.get("shares")

    if numeric_value_missing("trailingEps"):
        try:
            income_statement = instrument.get_income_stmt(freq="trailing")
            if isinstance(income_statement, pd.DataFrame):
                for row_name in ("DilutedEPS", "BasicEPS"):
                    if row_name not in income_statement.index:
                        continue
                    available_eps = pd.to_numeric(
                        income_statement.loc[row_name],
                        errors="coerce",
                    ).dropna()
                    if not available_eps.empty:
                        market_info["trailingEps"] = float(
                            available_eps.iloc[0]
                        )
                        break
        except Exception:
            pass

    if numeric_value_missing("trailingPE"):
        latest_price = _available_number(fast_info.get("lastPrice"))
        trailing_eps = _available_number(market_info.get("trailingEps"))
        if latest_price is not None and trailing_eps not in {None, 0}:
            market_info["trailingPE"] = latest_price / trailing_eps

    if numeric_value_missing("dividendRate"):
        try:
            dividend_rate = estimate_forward_dividend_rate(
                instrument.get_dividends(period="1y")
            )
            if dividend_rate is not None:
                market_info["dividendRate"] = dividend_rate
        except Exception:
            pass

    if not _format_market_date(market_info.get("exDividendDate")):
        try:
            calendar = instrument.get_calendar()
            if isinstance(calendar, Mapping):
                market_info["exDividendDate"] = calendar.get(
                    "Ex-Dividend Date"
                )
        except Exception:
            pass

    return market_info


@st.cache_data(ttl=FUNDAMENTALS_CACHE_SECONDS, show_spinner=False)
def get_fundamental_metrics(ticker: str) -> list[tuple[str, str]]:
    """Fetch and format available company fundamentals with fallbacks."""
    return format_fundamental_metrics(collect_fundamental_info(ticker))


@st.cache_data(ttl=300, show_spinner=False)
def download_watchlist_snapshot(ticker: str) -> tuple[float, float]:
    """Return the latest price and one-day change for a watchlist ticker."""
    data = yf.download(
        ticker,
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if data.empty or "Close" not in data.columns:
        raise ValueError("Yahoo Finance returned no recent closing prices.")
    return latest_daily_change(data["Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_details(ticker: str) -> tuple[str, str]:
    """Return a readable market name and exchange for one symbol."""
    market_name = None
    exchange_name = None

    try:
        search_results = yf.Search(
            ticker,
            max_results=8,
            news_count=0,
            lists_count=0,
        ).quotes

        exact_match = next(
            (
                result
                for result in search_results
                if result.get("symbol", "").upper() == ticker
            ),
            None,
        )
        if exact_match:
            market_name = exact_match.get("longname") or exact_match.get(
                "shortname"
            )
            exchange_name = exact_match.get("exchDisp") or exact_match.get(
                "exchange"
            )
    except Exception:
        pass

    if not market_name or not exchange_name:
        try:
            market_info = yf.Ticker(ticker).get_info()
            market_name = market_name or market_info.get(
                "longName"
            ) or market_info.get("shortName")
            exchange_name = exchange_name or market_info.get(
                "fullExchangeName"
            ) or market_info.get("exchange")
        except Exception:
            pass

    if not market_name:
        # Exceptions are not cached, so a temporary Yahoo failure can recover
        # automatically on the next Streamlit rerun.
        raise ValueError("Yahoo Finance returned no readable market name.")

    return market_name, readable_exchange_name(exchange_name)


def fallback_market_details(ticker: str) -> tuple[str, str]:
    """Provide a clean title while Yahoo's metadata service is unavailable."""
    return COMMON_MARKETS.get(ticker, ("Market", "MARKET"))


def build_close_figure(
    stock_data: pd.DataFrame,
    momentum_return: float | None = None,
):
    """Create a close chart coloured by the newest one-day momentum."""
    if momentum_return is None:
        momentum_return = latest_market_summary(stock_data)["return"]
    direction = momentum_direction(momentum_return)
    figure = px.line(
        stock_data,
        x=stock_data.index,
        y="Close",
        markers=False,
        labels={"Date": "Date", "Close": "Closing value (USD)"},
    )
    figure.update_traces(
        line={"color": MOMENTUM_COLOURS[direction], "width": 2.5},
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Close: %{y:,.2f} USD"
            "<extra></extra>"
        ),
    )
    figure.update_layout(hovermode="closest", showlegend=False)
    return figure


def build_watchlist_table(
    snapshots: list[tuple[str, float | None, float | None]],
) -> str:
    """Build a compact, non-wrapping watchlist table from price snapshots."""
    table_rows = []
    for ticker, current_price, percentage_change in snapshots:
        if current_price is None or percentage_change is None:
            price_text = "—"
            change_text = "—"
            change_class = "unavailable"
        else:
            price_text = f"${current_price:,.2f}"
            change_text = f"{percentage_change:+.2f}%"
            change_class = (
                "positive"
                if percentage_change > 0
                else "negative" if percentage_change < 0 else "neutral"
            )

        ticker_query = urlencode({"ticker": ticker})
        safe_ticker = escape(ticker)
        table_rows.append(
            "<tr>"
            '<td class="ticker">'
            f'<a class="ticker-link" href="?{ticker_query}" target="_self" '
            f'aria-label="Show {safe_ticker} market trend">{safe_ticker}</a>'
            "</td>"
            f'<td class="price {change_class}">{price_text}</td>'
            f'<td class="change {change_class}">{change_text}</td>'
            "</tr>"
        )

    return f"""
        <style>
        .watchlist-table {{
            border: none !important;
            border-collapse: collapse;
            font-size: 0.78rem;
            table-layout: fixed;
            width: 100%;
        }}
        .watchlist-table th,
        .watchlist-table td {{
            border: none !important;
            overflow: hidden;
            padding: 0.5rem 0.08rem;
            text-overflow: clip;
            white-space: nowrap;
        }}
        .watchlist-table th {{
            color: #8b919d;
            font-weight: 500;
            text-align: right;
        }}
        .watchlist-table .ticker {{
            font-weight: 700;
            text-align: left;
            width: 34%;
        }}
        .watchlist-table .price {{
            font-weight: 600;
            text-align: right;
            width: 36%;
        }}
        .watchlist-table .change {{
            font-weight: 600;
            text-align: right;
            width: 30%;
        }}
        .watchlist-table .ticker-link {{
            color: inherit;
            text-decoration: none;
        }}
        .watchlist-table .ticker-link:hover,
        .watchlist-table .ticker-link:focus-visible {{
            text-decoration: underline;
        }}
        .watchlist-table .positive {{ color: #22c55e; }}
        .watchlist-table .negative {{ color: #ef6461; }}
        .watchlist-table .neutral,
        .watchlist-table .unavailable {{ color: #8b919d; }}
        .watchlist-caption {{
            height: 1px;
            margin: -1px;
            overflow: hidden;
            padding: 0;
            position: absolute;
            width: 1px;
        }}
        </style>
        <table class="watchlist-table">
            <caption class="watchlist-caption">
                Saved tickers in order of addition
            </caption>
            <colgroup>
                <col style="width: 34%">
                <col style="width: 36%">
                <col style="width: 30%">
            </colgroup>
            <thead>
                <tr>
                    <th scope="col" style="text-align: left">Ticker</th>
                    <th scope="col">Price</th>
                    <th scope="col">1D</th>
                </tr>
            </thead>
            <tbody>{''.join(table_rows)}</tbody>
        </table>
    """


def render_watchlist(
    watchlist: list[str], storage_available: bool = True
) -> None:
    """Render a compact ordered watchlist in the sidebar."""
    with st.sidebar.container(height=390, border=True):
        st.subheader("Watchlist")
        notice = st.session_state.pop("watchlist_notice", None)
        if notice:
            st.success(notice)
        if not watchlist:
            st.caption("Save a ticker to start your watchlist.")
            return

        snapshots = []
        for watchlist_ticker in watchlist:
            try:
                current_price, percentage_change = download_watchlist_snapshot(
                    watchlist_ticker
                )
            except Exception:
                current_price = None
                percentage_change = None
            snapshots.append(
                (watchlist_ticker, current_price, percentage_change)
            )

        st.markdown(
            build_watchlist_table(snapshots),
            unsafe_allow_html=True,
        )

        ticker_to_remove = st.selectbox(
            "Ticker to delete",
            options=watchlist,
            key="watchlist_ticker_to_delete",
            help="Choose the ticker you want to remove from your watchlist.",
        )
        delete_ticker = st.button(
            "Delete selected ticker",
            use_container_width=True,
            disabled=not storage_available,
        )
        if delete_ticker:
            try:
                if remove_ticker_from_watchlist(ticker_to_remove):
                    st.session_state["watchlist_notice"] = (
                        f"{ticker_to_remove} was removed from your watchlist."
                    )
                    st.rerun()
            except ValueError as error:
                st.warning(str(error))
            except sqlite3.Error:
                st.warning("The ticker could not be removed locally.")


def render_market_summary(stock_data: pd.DataFrame) -> None:
    """Render the newest available daily market prices."""
    try:
        summary = latest_market_summary(stock_data)
    except ValueError:
        st.info("Latest daily market values are unavailable for this symbol.")
        return
    st.subheader("Latest Market Data")
    st.caption(
        "Newest available trading day: "
        f"{summary['date'].date().isoformat()}"
    )

    return_direction = momentum_direction(summary["return"])
    latest_values = [
        (
            "Open",
            "—" if summary["open"] is None else f"${summary['open']:,.2f}",
            "neutral",
        ),
        (
            "High",
            "—" if summary["high"] is None else f"${summary['high']:,.2f}",
            "neutral",
        ),
        (
            "Low",
            "—" if summary["low"] is None else f"${summary['low']:,.2f}",
            "neutral",
        ),
        (
            "Return",
            (
                "—"
                if summary["return"] is None
                else f"{summary['return']:+.2f}%"
            ),
            return_direction,
        ),
        (
            "Volume",
            (
                "—"
                if summary["volume"] is None
                else _format_compact_number(summary["volume"])
            ),
            "neutral",
        ),
        (
            "Moving Average 50D",
            (
                "—"
                if summary["ma_50"] is None
                else f"${summary['ma_50']:,.2f}"
            ),
            "neutral",
        ),
        (
            "Moving Average 200D",
            (
                "—"
                if summary["ma_200"] is None
                else f"${summary['ma_200']:,.2f}"
            ),
            "neutral",
        ),
        (
            "RSI (14D)",
            "—" if summary["rsi"] is None else f"{summary['rsi']:,.2f}",
            "neutral",
        ),
    ]
    st.markdown(
        build_metric_grid(
            latest_values,
            layout="latest",
            accessible_name="Latest market data",
        ),
        unsafe_allow_html=True,
    )


def render_fundamentals(ticker: str) -> None:
    """Render company fundamentals or an explicit availability message."""
    st.subheader("Company Fundamentals")
    try:
        metrics = get_fundamental_metrics(ticker)
    except Exception:
        metrics = []
    if not metrics:
        st.info(
            "Company fundamentals are temporarily unavailable from Yahoo "
            "Finance."
        )
        return

    st.markdown(
        build_metric_grid(
            [
                (label, value, "neutral")
                for label, value in metrics
            ],
            layout="fundamentals",
            accessible_name="Company fundamentals",
        ),
        unsafe_allow_html=True,
    )


def _safe_yahoo_news_url(value: object) -> str | None:
    """Allow only HTTPS Yahoo Finance article links."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    try:
        parsed_url = urlparse(candidate)
        port = parsed_url.port
    except ValueError:
        return None
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != "finance.yahoo.com"
        or parsed_url.username is not None
        or parsed_url.password is not None
        or port not in {None, 443}
    ):
        return None
    return candidate


def _compact_news_text(value: object, maximum_length: int) -> str:
    """Normalize and bound text received from Yahoo's news feed."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:maximum_length]


def _news_timestamp(value: object) -> pd.Timestamp | None:
    """Normalize epoch or ISO publication times to UTC."""
    try:
        if isinstance(value, (int, float)):
            timestamp = pd.to_datetime(value, unit="s", utc=True)
        else:
            timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def normalize_news_items(
    raw_news: object,
    ticker: str,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Return the newest relevant, display-safe Yahoo Finance stories."""
    if not isinstance(raw_news, list):
        return []

    normalized_ticker = clean_ticker(ticker)
    candidates: list[tuple[pd.Timestamp, dict[str, str]]] = []
    seen_urls: set[str] = set()

    for raw_item in raw_news:
        if not isinstance(raw_item, Mapping):
            continue
        nested_content = raw_item.get("content")
        content = (
            nested_content if isinstance(nested_content, Mapping) else raw_item
        )

        related_tickers = raw_item.get("relatedTickers")
        if isinstance(related_tickers, (list, tuple, set)):
            normalized_relations = {
                clean_ticker(value)
                for value in related_tickers
                if isinstance(value, str)
            }
            if (
                normalized_relations
                and normalized_ticker not in normalized_relations
            ):
                continue

        title = _compact_news_text(content.get("title"), 180)
        provider_data = content.get("provider")
        if isinstance(provider_data, Mapping):
            publisher_value = provider_data.get("displayName")
        else:
            publisher_value = content.get("publisher")
        publisher = _compact_news_text(publisher_value, 80) or "Yahoo Finance"

        publication_value = content.get("pubDate")
        if publication_value is None:
            publication_value = content.get("providerPublishTime")
        publication_time = _news_timestamp(publication_value)

        url_candidates: list[object] = [content.get("link")]
        for key in ("clickThroughUrl", "canonicalUrl"):
            url_data = content.get(key)
            if isinstance(url_data, Mapping):
                url_candidates.append(url_data.get("url"))
        article_url = next(
            (
                safe_url
                for value in url_candidates
                if (safe_url := _safe_yahoo_news_url(value)) is not None
            ),
            None,
        )

        if (
            not title
            or publication_time is None
            or article_url is None
            or article_url in seen_urls
        ):
            continue
        seen_urls.add(article_url)
        candidates.append(
            (
                publication_time,
                {
                    "title": title,
                    "publisher": publisher,
                    "published": publication_time.strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                    "url": article_url,
                },
            )
        )

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    safe_limit = max(0, min(int(limit), 10))
    return [item for _, item in candidates[:safe_limit]]


def build_news_list(news_items: list[dict[str, str]]) -> str:
    """Build an accessible compact list from normalized news stories."""
    story_rows = []
    for item in news_items:
        story_rows.append(
            '<li class="news-item">'
            f'<a class="news-link" href="{escape(item["url"], quote=True)}" '
            'target="_blank" rel="noopener noreferrer">'
            f'{escape(item["title"])}'
            "</a>"
            '<div class="news-meta">'
            f'{escape(item["publisher"])} · {escape(item["published"])}'
            "</div>"
            "</li>"
        )
    return (
        '<ul class="news-list" aria-label="Latest Yahoo Finance news">'
        f'{"".join(story_rows)}'
        "</ul>"
    )


@st.cache_data(ttl=900, show_spinner=False)
def get_latest_news(ticker: str) -> list[dict[str, str]]:
    """Fetch and normalize the newest Yahoo Finance ticker stories."""
    search_result = yf.Search(
        ticker,
        max_results=1,
        news_count=12,
        lists_count=0,
        include_cb=False,
        include_nav_links=False,
        include_research=False,
        include_cultural_assets=False,
        recommended=0,
    )
    return normalize_news_items(search_result.news, ticker, limit=3)


def render_latest_news(ticker: str) -> None:
    """Render the three newest Yahoo Finance stories for a ticker."""
    st.subheader("Latest News")
    try:
        news_items = get_latest_news(ticker)
    except Exception:
        news_items = []
    if not news_items:
        st.info("No recent Yahoo Finance news is available for this symbol.")
        return
    st.markdown(build_news_list(news_items), unsafe_allow_html=True)


def main() -> None:
    """Display the Streamlit dashboard."""
    st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")

    st.markdown(
        """
        <style>
        div[data-baseweb="tooltip"] > div {
            background-color: rgba(14, 17, 23, 0.30) !important;
            backdrop-filter: blur(6px);
        }

        p.market-identity {
            color: #8b919d;
            font-size: 1.5rem !important;
            letter-spacing: 0.02em;
            margin-top: -0.75rem;
            margin-bottom: 1.5rem;
        }
        """
        + METRIC_GRID_STYLES
        + NEWS_STYLES
        + """
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Market search")
    query_ticker = selected_ticker_from_query(st.query_params)
    previous_query_ticker = st.session_state.get("watchlist_query_ticker")
    if "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = query_ticker
    elif query_ticker and query_ticker != previous_query_ticker:
        st.session_state["ticker_input"] = query_ticker
    if query_ticker:
        st.session_state["watchlist_query_ticker"] = query_ticker

    ticker = clean_ticker(
        st.sidebar.text_input(
            "Enter a stock, index, or commodity symbol",
            key="ticker_input",
            placeholder="Type a market symbol",
            help=(
                "Enter the exact Yahoo Finance symbol. Examples: AAPL (stock), "
                "^GSPC (index), or GC=F (commodity)."
            ),
            on_change=clear_selected_ticker_query,
        )
    )

    start_date = st.sidebar.date_input(
        "Start date",
        value=date.today() - timedelta(days=365),
        max_value=date.today(),
        help=(
            "Choose the first date shown in the chart. The end date always "
            "uses the newest available Yahoo Finance trading day."
        ),
    )
    st.sidebar.caption("End date: newest available trading day.")

    try:
        watchlist = load_watchlist()
        watchlist_storage_available = True
    except sqlite3.Error:
        watchlist = []
        watchlist_storage_available = False
        st.sidebar.warning("The local watchlist storage is unavailable.")

    add_to_watchlist = st.sidebar.button(
        "Add to watchlist",
        type="primary",
        use_container_width=True,
        disabled=not ticker or not watchlist_storage_available,
    )
    if add_to_watchlist:
        try:
            if add_ticker_to_watchlist(ticker):
                watchlist.append(validate_ticker(ticker))
                st.sidebar.success(f"{ticker} was added to your watchlist.")
            else:
                st.sidebar.info(f"{ticker} is already in your watchlist.")
        except ValueError as error:
            st.sidebar.warning(str(error))
        except sqlite3.Error:
            st.sidebar.warning("The ticker could not be saved locally.")

    render_watchlist(watchlist, watchlist_storage_available)
    st.sidebar.caption("Saved locally on this app in order of addition.")

    if not ticker:
        st.title("Financial Market Dashboard")
        st.warning("Enter a valid stock, index, or commodity symbol to begin.")
        return

    try:
        with st.spinner(f"Downloading {ticker} data..."):
            complete_stock_data = add_technical_indicators(
                download_stock_data(ticker, start_date.isoformat())
            )
            stock_data = filter_data_from_start(
                complete_stock_data,
                start_date.isoformat(),
            )
    except ValueError:
        st.title("Financial Market Dashboard")
        st.warning(
            f"No recent data was found for {ticker}. Check the symbol, then "
            "try again."
        )
        return
    except Exception:
        st.title("Financial Market Dashboard")
        st.warning(
            "The data could not be downloaded. Check the symbol and your "
            "internet connection, then try again."
        )
        return

    if stock_data.empty or "Close" not in stock_data.columns:
        st.title("Financial Market Dashboard")
        st.warning(
            f"No recent data was found for {ticker}. Check the symbol, then "
            "try again."
        )
        return

    try:
        market_name, exchange_name = get_market_details(ticker)
    except Exception:
        market_name, exchange_name = fallback_market_details(ticker)

    market_title = (
        "Market Trend" if market_name == "Market" else f"{market_name} Market Trend"
    )
    st.title(market_title)
    st.markdown(
        f'<p class="market-identity">{escape(exchange_name)}: '
        f"{escape(ticker)}</p>",
        unsafe_allow_html=True,
    )

    latest_summary = latest_market_summary(complete_stock_data)
    figure = build_close_figure(
        stock_data,
        momentum_return=latest_summary["return"],
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False},
    )

    render_market_summary(complete_stock_data)
    render_fundamentals(ticker)
    render_latest_news(ticker)

    csv_data = stock_data.to_csv().encode("utf-8")
    safe_ticker = ticker.replace("^", "").replace("=", "_")
    st.download_button(
        label="Download current data as CSV",
        data=csv_data,
        file_name=f"{safe_ticker}_{start_date.isoformat()}_to_latest.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
