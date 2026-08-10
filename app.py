"""A simple interactive stock-price dashboard built with Streamlit."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


def clean_ticker(ticker: str) -> str:
    """Remove extra spaces and use the uppercase ticker format."""
    return ticker.strip().upper()


def dates_are_valid(start_date: date | str, end_date: date | str) -> bool:
    """Return True when the start date is not after the end date."""
    return pd.Timestamp(start_date) <= pd.Timestamp(end_date)


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

    data.index.name = "Date"
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_name(ticker: str) -> str:
    """Return the company, index, or commodity name for a market symbol."""
    try:
        market_info = yf.Ticker(ticker).get_info()
    except Exception:
        return "Selected Market"

    return (
        market_info.get("longName")
        or market_info.get("shortName")
        or "Selected Market"
    )


def main() -> None:
    """Display the Streamlit dashboard."""
    st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")

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

    market_name = get_market_name(ticker)
    st.title(f"{market_name} Market Trend")

    st.header("Closing-value trend")
    figure = px.line(
        stock_data,
        x=stock_data.index,
        y="Close",
        labels={"Date": "Date", "Close": "Closing value"},
        title=f"{market_name} closing-value trend",
    )
    figure.update_layout(hovermode="x unified")
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
