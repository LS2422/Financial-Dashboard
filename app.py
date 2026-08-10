from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Mapping
from html import escape
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


WATCHLIST_DB_PATH = Path(__file__).with_name(".watchlist.db")
TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9^.=-]{0,19}$")

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


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker: str) -> pd.DataFrame:
    """Download one ticker's trailing year and cache it for one hour."""
    data = yf.download(
        ticker,
        period="1y",
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


def latest_market_summary(stock_data: pd.DataFrame) -> dict[str, object]:
    """Return the newest available OHLC values and one-day close return."""
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
        "return": daily_return,
    }


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


def format_fundamental_metrics(
    market_info: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Format only the company fundamentals Yahoo currently provides."""
    metrics: list[tuple[str, str]] = []

    trailing_pe = _available_number(market_info.get("trailingPE"))
    if trailing_pe is not None:
        metrics.append(("P/E Ratio", f"{trailing_pe:,.2f}"))

    dividend_parts = []
    dividend_rate = _available_number(market_info.get("dividendRate"))
    if dividend_rate is not None:
        dividend_parts.append(f"${dividend_rate:,.2f}")

    dividend_yield = _available_number(market_info.get("dividendYield"))
    if dividend_yield is not None:
        dividend_parts.append(f"{dividend_yield:,.2f}%")
    if dividend_parts:
        metrics.append(("Dividend", " · ".join(dividend_parts)))

    latest_dividend = _available_number(market_info.get("lastDividendValue"))
    if latest_dividend is not None:
        metrics.append(("Quarterly Dividend", f"${latest_dividend:,.2f}"))

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


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_metrics(ticker: str) -> list[tuple[str, str]]:
    """Fetch and format available company fundamentals for one ticker."""
    market_info = yf.Ticker(ticker).get_info()
    if not isinstance(market_info, Mapping):
        return []
    return format_fundamental_metrics(market_info)


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


def build_close_figure(stock_data: pd.DataFrame):
    """Create the closing-value chart without persistent point markers."""
    figure = px.line(
        stock_data,
        x=stock_data.index,
        y="Close",
        markers=False,
        labels={"Date": "Date", "Close": "Closing value (USD)"},
    )
    figure.update_traces(
        line={"color": "#22c55e", "width": 2.5},
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

    latest_values = [
        (
            "Open",
            "—" if summary["open"] is None else f"${summary['open']:,.2f}",
        ),
        (
            "High",
            "—" if summary["high"] is None else f"${summary['high']:,.2f}",
        ),
        (
            "Low",
            "—" if summary["low"] is None else f"${summary['low']:,.2f}",
        ),
        (
            "Return",
            (
                "—"
                if summary["return"] is None
                else f"{summary['return']:+.2f}%"
            ),
        ),
    ]
    for column, (label, value) in zip(st.columns(4), latest_values):
        column.metric(label, value)


def render_fundamentals(ticker: str) -> None:
    """Render company fundamentals only when Yahoo provides them."""
    try:
        metrics = get_fundamental_metrics(ticker)
    except Exception:
        metrics = []
    if not metrics:
        return

    st.subheader("Company Fundamentals")
    for start_index in range(0, len(metrics), 4):
        metric_group = metrics[start_index : start_index + 4]
        for column, (label, value) in zip(
            st.columns(len(metric_group)), metric_group
        ):
            column.metric(label, value)


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
            stock_data = download_stock_data(ticker)
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

    figure = build_close_figure(stock_data)
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False},
    )

    render_market_summary(stock_data)
    render_fundamentals(ticker)

    csv_data = stock_data.to_csv().encode("utf-8")
    safe_ticker = ticker.replace("^", "").replace("=", "_")
    st.download_button(
        label="Download current data as CSV",
        data=csv_data,
        file_name=f"{safe_ticker}_trailing_year.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
