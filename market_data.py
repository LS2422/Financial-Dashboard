"""Reusable market-data contracts and the Yahoo Finance implementation."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import yfinance as yf

from app_logging import get_logger, log_event


TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9^.=-]{0,19}$")
YAHOO_NETWORK_RETRIES = 2
yf.config.network.retries = YAHOO_NETWORK_RETRIES
LOGGER = get_logger(__name__)


def clean_ticker(ticker: str) -> str:
    """Remove extra spaces and use Yahoo's uppercase ticker format."""
    return ticker.strip().upper()


def validate_ticker(ticker: str) -> str:
    """Return a normalized ticker or raise when its format is unsafe."""
    normalized_ticker = clean_ticker(ticker)
    if not TICKER_PATTERN.fullmatch(normalized_ticker):
        raise ValueError("Enter a valid ticker symbol.")
    return normalized_ticker


def _available_number(value: object) -> float | None:
    """Return a finite number or None for unavailable Yahoo metadata."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _has_valid_market_date(value: object) -> bool:
    """Return whether Yahoo supplied a usable date-like value."""
    try:
        if isinstance(value, (int, float)):
            timestamp = pd.to_datetime(value, unit="s", utc=True)
        else:
            timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError, OverflowError):
        return False
    return not pd.isna(timestamp)


def estimate_forward_dividend_rate(dividends: pd.Series) -> float | None:
    """Estimate an annual rate from the latest recurring dividend payment."""
    if not isinstance(dividends, pd.Series) or dividends.empty:
        return None

    available_dividends = pd.to_numeric(dividends, errors="coerce").dropna()
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


@dataclass(frozen=True)
class MarketDetails:
    """Display-ready company/index name and its raw exchange name."""

    name: str
    exchange: str | None


@runtime_checkable
class MarketDataSource(Protocol):
    """The market information required by the dashboard."""

    def history(
        self,
        ticker: str,
        *,
        start_date: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame: ...

    def quote_type(self, ticker: str) -> str: ...

    def market_details(self, ticker: str) -> MarketDetails: ...

    def fundamentals(self, ticker: str) -> dict[str, object]: ...

    def news(
        self,
        ticker: str,
        *,
        count: int = 12,
    ) -> list[Mapping[str, object]]: ...


class YahooFinanceDataSource:
    """Adapt yfinance responses to the dashboard's stable interface."""

    def __init__(self, yahoo_module: Any = yf) -> None:
        self.yahoo = yahoo_module

    def history(
        self,
        ticker: str,
        *,
        start_date: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        normalized_ticker = validate_ticker(ticker)
        if (start_date is None) == (period is None):
            raise ValueError("Provide exactly one of start_date or period.")

        request_range = (
            {"start": start_date}
            if start_date is not None
            else {"period": period}
        )
        try:
            history = self.yahoo.download(
                normalized_ticker,
                **request_range,
                interval="1d",
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
        except Exception:
            log_event(
                LOGGER,
                logging.ERROR,
                "yahoo_history_failed",
                ticker=normalized_ticker,
                operation="history",
                exc_info=True,
            )
            raise

        if not isinstance(history, pd.DataFrame):
            log_event(
                LOGGER,
                logging.ERROR,
                "yahoo_history_invalid",
                ticker=normalized_ticker,
                operation="history",
                response_type=type(history).__name__,
            )
            raise ValueError("Yahoo Finance history must be a dataframe.")

        log_event(
            LOGGER,
            logging.INFO,
            "yahoo_history_downloaded",
            ticker=normalized_ticker,
            operation="history",
            rows=len(history),
        )
        return history

    def quote_type(self, ticker: str) -> str:
        normalized_ticker = validate_ticker(ticker)
        quote_type = None
        try:
            search_results = self.yahoo.Search(
                normalized_ticker,
                max_results=8,
                news_count=0,
                lists_count=0,
            ).quotes
            exact_match = next(
                (
                    result
                    for result in search_results
                    if result.get("symbol", "").upper() == normalized_ticker
                ),
                None,
            )
            if exact_match:
                quote_type = exact_match.get("quoteType")
        except Exception:
            log_event(
                LOGGER,
                logging.WARNING,
                "yahoo_quote_search_failed",
                ticker=normalized_ticker,
                operation="quote_type",
                exc_info=True,
            )

        if not quote_type:
            try:
                market_info = self.yahoo.Ticker(normalized_ticker).get_info()
                if isinstance(market_info, Mapping):
                    quote_type = market_info.get("quoteType")
            except Exception:
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "yahoo_quote_type_failed",
                    ticker=normalized_ticker,
                    operation="quote_type",
                    exc_info=True,
                )

        if not isinstance(quote_type, str) or not quote_type.strip():
            raise ValueError("Yahoo Finance returned no instrument type.")
        return quote_type.strip().upper()

    def market_details(self, ticker: str) -> MarketDetails:
        normalized_ticker = validate_ticker(ticker)
        market_name = None
        exchange_name = None
        try:
            search_results = self.yahoo.Search(
                normalized_ticker,
                max_results=8,
                news_count=0,
                lists_count=0,
            ).quotes
            exact_match = next(
                (
                    result
                    for result in search_results
                    if result.get("symbol", "").upper() == normalized_ticker
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
            log_event(
                LOGGER,
                logging.WARNING,
                "yahoo_market_search_failed",
                ticker=normalized_ticker,
                operation="market_details",
                exc_info=True,
            )

        if not market_name or not exchange_name:
            try:
                market_info = self.yahoo.Ticker(normalized_ticker).get_info()
                if isinstance(market_info, Mapping):
                    market_name = market_name or market_info.get(
                        "longName"
                    ) or market_info.get("shortName")
                    exchange_name = exchange_name or market_info.get(
                        "fullExchangeName"
                    ) or market_info.get("exchange")
            except Exception:
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "yahoo_market_details_failed",
                    ticker=normalized_ticker,
                    operation="market_details",
                    exc_info=True,
                )

        if not market_name:
            raise ValueError("Yahoo Finance returned no readable market name.")
        return MarketDetails(str(market_name), exchange_name)

    def fundamentals(self, ticker: str) -> dict[str, object]:
        """Collect fundamentals with independent fallbacks for Yahoo outages."""
        normalized_ticker = validate_ticker(ticker)
        instrument = self.yahoo.Ticker(normalized_ticker)
        market_info: dict[str, object] = {}
        detailed_info_loaded = False

        try:
            detailed_info = instrument.get_info()
            if isinstance(detailed_info, Mapping):
                market_info.update(detailed_info)
                detailed_info_loaded = bool(detailed_info)
        except Exception:
            log_event(
                LOGGER,
                logging.WARNING,
                "yahoo_fundamentals_fallback",
                ticker=normalized_ticker,
                operation="detailed_info",
                exc_info=True,
            )

        if not detailed_info_loaded:
            try:
                instrument = self.yahoo.Ticker(normalized_ticker)
                detailed_info = instrument.get_info()
                if isinstance(detailed_info, Mapping):
                    market_info.update(detailed_info)
            except Exception:
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "yahoo_fundamentals_fallback",
                    ticker=normalized_ticker,
                    operation="detailed_info_retry",
                    exc_info=True,
                )

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
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "yahoo_fundamentals_fallback",
                    ticker=normalized_ticker,
                    operation="fast_info",
                    exc_info=True,
                )

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
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "yahoo_fundamentals_fallback",
                    ticker=normalized_ticker,
                    operation="income_statement",
                    exc_info=True,
                )

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
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "yahoo_fundamentals_fallback",
                    ticker=normalized_ticker,
                    operation="dividends",
                    exc_info=True,
                )

        if not _has_valid_market_date(market_info.get("exDividendDate")):
            try:
                calendar = instrument.get_calendar()
                if isinstance(calendar, Mapping):
                    market_info["exDividendDate"] = calendar.get(
                        "Ex-Dividend Date"
                    )
            except Exception:
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "yahoo_fundamentals_fallback",
                    ticker=normalized_ticker,
                    operation="calendar",
                    exc_info=True,
                )

        log_event(
            LOGGER,
            logging.INFO,
            "yahoo_fundamentals_collected",
            ticker=normalized_ticker,
            operation="fundamentals",
            available_fields=sum(value is not None for value in market_info.values()),
        )
        return market_info

    def news(
        self,
        ticker: str,
        *,
        count: int = 12,
    ) -> list[Mapping[str, object]]:
        normalized_ticker = validate_ticker(ticker)
        try:
            search_result = self.yahoo.Search(
                normalized_ticker,
                max_results=1,
                news_count=count,
                lists_count=0,
                include_cb=False,
                include_nav_links=False,
                include_research=False,
                include_cultural_assets=False,
                recommended=0,
            )
            raw_news = search_result.news
        except Exception:
            log_event(
                LOGGER,
                logging.ERROR,
                "yahoo_news_failed",
                ticker=normalized_ticker,
                operation="news",
                exc_info=True,
            )
            raise

        if not isinstance(raw_news, list):
            log_event(
                LOGGER,
                logging.ERROR,
                "yahoo_news_invalid",
                ticker=normalized_ticker,
                operation="news",
                response_type=type(raw_news).__name__,
            )
            raise ValueError("Yahoo Finance news must be a list.")

        news_items = [item for item in raw_news if isinstance(item, Mapping)]
        log_event(
            LOGGER,
            logging.INFO,
            "yahoo_news_received",
            ticker=normalized_ticker,
            operation="news",
            stories=len(news_items),
        )
        return news_items


YAHOO_MARKET_DATA_SOURCE: MarketDataSource = YahooFinanceDataSource()
