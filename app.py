from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from datetime import date, timedelta
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


def dates_are_valid(start_date: date | str, end_date: date | str) -> bool:
    """Return True when the start date is not after the end date."""
    return pd.Timestamp(start_date) <= pd.Timestamp(end_date)


def readable_exchange_name(exchange: str | None) -> str:
    """Convert Yahoo's exchange code into a readable display name."""
    if not exchange:
        return "MARKET"

    cleaned_exchange = exchange.strip().upper()
    return EXCHANGE_NAMES.get(cleaned_exchange, cleaned_exchange)


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Download one ticker's daily prices and cache them for one hour."""
    # yfinance treats its end date as exclusive, so add one day to include the
    # date chosen by the user.
    inclusive_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    data = yf.download(
        ticker,
        start=start_date,
        end=inclusive_end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    # Raising prevents Streamlit from caching a temporary empty Yahoo response.
    if data.empty or "Close" not in data.columns:
        raise ValueError("Yahoo Finance returned no closing-price data.")

    data.index.name = "Date"
    return data


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


def render_watchlist(watchlist: list[str]) -> None:
    """Render a compact ordered watchlist in the sidebar."""
    with st.sidebar.container(height=300, border=True):
        st.subheader("Watchlist")
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

    default_end_date = date.today()
    default_start_date = default_end_date - timedelta(days=365)
    start_date = st.sidebar.date_input("Start date", value=default_start_date)
    end_date = st.sidebar.date_input("End date", value=default_end_date)

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

    render_watchlist(watchlist)
    st.sidebar.caption("Saved locally on this app in order of addition.")

    if not ticker:
        st.title("Financial Market Dashboard")
        st.warning("Enter a valid stock, index, or commodity symbol to begin.")
        return

    if not dates_are_valid(start_date, end_date):
        st.title("Financial Market Dashboard")
        st.warning("The start date must be on or before the end date.")
        return

    try:
        with st.spinner(f"Downloading {ticker} data..."):
            stock_data = download_stock_data(
                ticker,
                start_date.isoformat(),
                end_date.isoformat(),
            )
    except ValueError:
        st.title("Financial Market Dashboard")
        st.warning(
            f"No data was found for {ticker} in this date range. "
            "Check the symbol and dates, then try again."
        )
        return
    except Exception:
        st.title("Financial Market Dashboard")
        st.warning(
            "The data could not be downloaded. Check the symbol, date range, "
            "and your internet connection, then try again."
        )
        return

    if stock_data.empty or "Close" not in stock_data.columns:
        st.title("Financial Market Dashboard")
        st.warning(
            f"No data was found for {ticker} in this date range. "
            "Check the symbol and dates, then try again."
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

    csv_data = stock_data.to_csv().encode("utf-8")
    safe_ticker = ticker.replace("^", "").replace("=", "_")
    st.download_button(
        label="Download current data as CSV",
        data=csv_data,
        file_name=f"{safe_ticker}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
