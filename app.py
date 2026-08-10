from __future__ import annotations

from datetime import date, timedelta
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


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
            font-size: 1rem !important;
            letter-spacing: 0.02em;
            margin-top: -0.75rem;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Market search")
    ticker = clean_ticker(
        st.sidebar.text_input(
            "Enter a stock, index, or commodity symbol",
            placeholder="Type a market symbol",
            help=(
                "Enter the exact Yahoo Finance symbol. Examples: AAPL (stock), "
                "^GSPC (index), or GC=F (commodity)."
            ),
        )
    )

    default_end_date = date.today()
    default_start_date = default_end_date - timedelta(days=365)
    start_date = st.sidebar.date_input("Start date", value=default_start_date)
    end_date = st.sidebar.date_input("End date", value=default_end_date)

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
