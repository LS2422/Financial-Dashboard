from __future__ import annotations

import math
import logging
from datetime import date

import pandas as pd
import streamlit as st

from app_logging import get_logger, log_event
from market_data import MarketDataSource, YAHOO_MARKET_DATA_SOURCE
import ml_prediction


PREDICTION_CACHE_SECONDS = 3600
EQUITY_QUOTE_TYPE = "EQUITY"
MARKET_DATA_SOURCE: MarketDataSource = YAHOO_MARKET_DATA_SOURCE
LOGGER = get_logger(__name__)


@st.cache_data(ttl=PREDICTION_CACHE_SECONDS, show_spinner=False)
def download_prediction_history(ticker: str, start_date: str) -> pd.DataFrame:
    """Download the independent history required by the ML artifact."""
    data = MARKET_DATA_SOURCE.history(
        ticker,
        start_date=start_date,
    )
    missing_columns = [
        column
        for column in ml_prediction.RAW_FEATURE_COLUMNS
        if column not in data.columns
    ]
    if data.empty or missing_columns:
        raise ValueError("Yahoo Finance returned incomplete required price data.")

    data.index.name = "Date"
    return data


@st.cache_resource(show_spinner=False)
def get_prediction_model_bundle() -> ml_prediction.ModelBundle:
    """Load the fixed evaluation model and latest production model once."""
    return ml_prediction.load_model_bundle()


def is_equity_quote_type(quote_type: str | None) -> bool:
    """Return whether Yahoo identifies the symbol as a listed equity."""
    return isinstance(quote_type, str) and quote_type.upper() == EQUITY_QUOTE_TYPE


@st.cache_data(ttl=PREDICTION_CACHE_SECONDS, show_spinner=False)
def get_instrument_quote_type(ticker: str) -> str:
    """Return Yahoo's quote type using search metadata with an info fallback."""
    return MARKET_DATA_SOURCE.quote_type(ticker)


def _render_prediction_metrics(
    forecast: ml_prediction.CurrentForecast,
    evaluation: ml_prediction.HistoricalEvaluation | None,
) -> None:
    """Render forecast outputs and fixed-window return metrics."""
    metric_columns = st.columns(4)
    metric_values: list[tuple[str, str, str]] = [
        (
            "Predicted next log return",
            f"{forecast.predicted_log_return * 100:+.3f}%",
            "The model directly predicts log(Adj Close[t+1] / Adj Close[t]).",
        ),
        (
            "Implied next adjusted close",
            f"${forecast.implied_next_adjusted_close:,.2f}",
            "Derived as the current adjusted close multiplied by exp(predicted log return).",
        ),
    ]
    if evaluation is not None:
        metric_values.extend(
            [
                (
                    "Log-return RMSE",
                    f"{evaluation.model.rmse * 100:.3f} pp",
                    "Root mean squared error on next-session log returns. One percentage point equals 0.01 in raw return units.",
                ),
                (
                    "Log-return R²",
                    (
                        f"{evaluation.model.r2:.3f}"
                        if math.isfinite(evaluation.model.r2)
                        else "—"
                    ),
                    "R² is calculated only from next-session log returns in the frozen test window.",
                ),
            ]
        )

    for metric_column, (label, value, help_text) in zip(
        metric_columns,
        metric_values,
    ):
        with metric_column:
            st.metric(label, value, help=help_text)


def _render_unseen_ticker_context(metadata: ml_prediction.ModelMetadata) -> None:
    """Show the frozen nested leave-one-ticker-out aggregate evidence."""
    summary = metadata.unseen_ticker_summary
    required_values = (
        "held_out_tickers",
        "observations",
        "pooled_rmse",
        "pooled_r2",
    )
    if not all(value in summary for value in required_values):
        return

    st.caption(
        "Broader unseen-stock context: nested leave-one-ticker-out testing "
        f"across {int(summary['held_out_tickers']):,} held-out tickers and "
        f"{int(summary['observations']):,} observations produced pooled "
        f"log-return RMSE {float(summary['pooled_rmse']) * 100:.3f} pp and "
        f"R² {float(summary['pooled_r2']):.3f}. This is universe-level "
        "context, not this ticker's own result."
    )


def render_ml_prediction(ticker: str, *, today: date | None = None) -> None:
    """Render an on-demand next-session return forecast for an equity."""
    st.subheader("ML Return Forecast")
    st.caption(
        "Experimental one-session forecast using the Task 7 engineered "
        "features. The fitted target is return, not price."
    )
    if not st.button(
        "Predict next trading day",
        key=f"predict_next_trading_day_{ticker}",
        type="primary",
    ):
        return

    try:
        quote_type = get_instrument_quote_type(ticker)
    except Exception as error:
        log_event(
            LOGGER,
            logging.WARNING,
            "ml_quote_type_unavailable",
            ticker=ticker,
            error_type=type(error).__name__,
            exc_info=True,
        )
        st.warning(
            "The forecast was not run because Yahoo Finance could not "
            "confirm that this symbol is a stock."
        )
        return
    if not is_equity_quote_type(quote_type):
        st.info(
            "The ML forecast is available only for stocks; Yahoo Finance "
            f"classifies this symbol as {quote_type.lower()}."
        )
        return

    try:
        bundle = get_prediction_model_bundle()
        history = download_prediction_history(
            ticker,
            bundle.metadata.feature_history_start,
        )
        forecast = ml_prediction.predict_next_return(
            history,
            bundle.production_model,
            bundle.metadata.feature_columns,
        )
        evaluation = ml_prediction.evaluate_ticker_history(
            history,
            bundle.evaluation_model,
            bundle.metadata,
        )
    except (FileNotFoundError, ImportError, ValueError) as error:
        log_event(
            LOGGER,
            logging.ERROR,
            "ml_forecast_unavailable",
            ticker=ticker,
            error_type=type(error).__name__,
            exc_info=True,
        )
        st.warning(f"The ML forecast is currently unavailable: {error}")
        return
    except Exception as error:
        log_event(
            LOGGER,
            logging.ERROR,
            "ml_forecast_failed",
            ticker=ticker,
            error_type=type(error).__name__,
            exc_info=True,
        )
        st.warning(
            "The ML forecast could not be produced from the latest Yahoo "
            "Finance data. Please try again later."
        )
        return

    metadata = bundle.metadata
    scope = ml_prediction.evaluation_scope(ticker, metadata)
    in_training_universe = ticker.upper() in set(metadata.training_tickers)

    _render_prediction_metrics(forecast, evaluation)
    st.caption(
        f"Forecast as of {forecast.as_of_date.date().isoformat()}; production "
        f"training-data end date {metadata.data_end_date}."
    )
    st.caption(
        f"Evaluation scope: {scope}. Frozen test window: "
        f"{metadata.test_start_date} to {metadata.test_end_date}. Usable "
        f"observations: {evaluation.observations if evaluation else 'unavailable'}."
    )
    st.caption(
        "Training-universe status: "
        f"{'inside' if in_training_universe else 'outside'} the approved "
        "18-stock large-cap universe."
    )

    if evaluation is not None:
        st.caption(
            "Zero-return baseline on the identical rows: log-return RMSE "
            f"{evaluation.zero_return_baseline.rmse * 100:.3f} pp; R² "
            f"{evaluation.zero_return_baseline.r2:.3f}."
        )
        if evaluation.model.rmse >= evaluation.zero_return_baseline.rmse:
            st.warning(
                "The model RMSE is not lower than the zero-return baseline "
                "on this ticker's frozen test rows."
            )
        if math.isfinite(evaluation.model.r2) and evaluation.model.r2 < 0:
            st.warning(
                "The model's log-return R² is negative on this ticker's "
                "frozen test rows."
            )
    else:
        st.info(
            "Ticker-specific RMSE and R² are unavailable because this symbol "
            "does not have enough usable observations in the frozen test "
            "window. The current forecast is still shown."
        )

    if not in_training_universe:
        st.warning(
            "This ticker is outside the 18-stock large-cap training universe. "
            "Treat the forecast as an extrapolation to a potentially "
            "different stock."
        )
        _render_unseen_ticker_context(metadata)

    if ml_prediction.is_artifact_stale(metadata, today=today):
        st.warning(
            "The production artifact may be stale: its training data ends "
            f"on {metadata.data_end_date}."
        )

    st.caption(
        "This experimental statistical forecast is not financial advice and "
        "does not represent a guaranteed future price."
    )
