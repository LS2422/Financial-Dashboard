from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]
ENGINEERED_FEATURE_COLUMNS = [
    "Log_Return",
    "Log_Return_Lag_1",
    "Log_Return_Lag_3",
    "Log_Return_Lag_5",
    "Price_Range",
    "Volume_Change",
    "Volatility_20",
    "MA_10_MA_20_Ratio",
    "Price_MA_20_Ratio",
    "RSI_14",
    "MACD_Histogram",
]
FEATURE_COLUMNS = [*RAW_FEATURE_COLUMNS, *ENGINEERED_FEATURE_COLUMNS]
MODEL_COLUMNS = FEATURE_COLUMNS
TARGET_COLUMN = "Target_Next_Log_Return"
DEFAULT_ARTIFACT_DIRECTORY = Path(__file__).with_name("model_artifacts")
STALE_AFTER_DAYS = 30


class ReturnModel(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return one next-session log-return prediction per row."""


@dataclass(frozen=True)
class ReturnMetrics:
    rmse: float
    r2: float
    mae: float
    directional_accuracy: float


@dataclass(frozen=True)
class HistoricalEvaluation:
    model: ReturnMetrics
    zero_return_baseline: ReturnMetrics
    observations: int
    first_evaluation_date: pd.Timestamp
    last_evaluation_date: pd.Timestamp


@dataclass(frozen=True)
class CurrentForecast:
    as_of_date: pd.Timestamp
    current_adjusted_close: float
    predicted_log_return: float
    implied_next_adjusted_close: float

    @property
    def predicted_percentage_return(self) -> float:
        return (math.exp(self.predicted_log_return) - 1) * 100


@dataclass(frozen=True)
class ModelMetadata:
    schema_version: int
    feature_columns: tuple[str, ...]
    target_column: str
    feature_history_start: str
    development_end_date: str
    test_start_date: str
    test_end_date: str
    data_end_date: str
    generated_at: str
    training_tickers: tuple[str, ...]
    minimum_evaluation_rows: int
    xgboost_version: str
    unseen_ticker_summary: dict[str, object]

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ModelMetadata":
        return cls(
            schema_version=int(values["schema_version"]),
            feature_columns=tuple(str(value) for value in values["feature_columns"]),
            target_column=str(values["target_column"]),
            feature_history_start=str(values["feature_history_start"]),
            development_end_date=str(values["development_end_date"]),
            test_start_date=str(values["test_start_date"]),
            test_end_date=str(values["test_end_date"]),
            data_end_date=str(values["data_end_date"]),
            generated_at=str(values["generated_at"]),
            training_tickers=tuple(
                str(value).upper() for value in values["training_tickers"]
            ),
            minimum_evaluation_rows=int(values["minimum_evaluation_rows"]),
            xgboost_version=str(values["xgboost_version"]),
            unseen_ticker_summary=dict(values.get("unseen_ticker_summary", {})),
        )

    @classmethod
    def for_tests(
        cls,
        *,
        test_start_date: str = "2025-02-01",
        test_end_date: str = "2025-03-31",
        data_end_date: str = "2026-08-01",
        training_tickers: Sequence[str] = ("MSFT", "NVDA"),
        minimum_evaluation_rows: int = 2,
    ) -> "ModelMetadata":
        return cls(
            schema_version=1,
            feature_columns=tuple(FEATURE_COLUMNS),
            target_column=TARGET_COLUMN,
            feature_history_start="2020-12-01",
            development_end_date="2025-01-31",
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            data_end_date=data_end_date,
            generated_at="2026-08-01T00:00:00+00:00",
            training_tickers=tuple(value.upper() for value in training_tickers),
            minimum_evaluation_rows=minimum_evaluation_rows,
            xgboost_version="2.1.4",
            unseen_ticker_summary={},
        )


@dataclass(frozen=True)
class ModelBundle:
    evaluation_model: ReturnModel
    production_model: ReturnModel
    metadata: ModelMetadata


def _prepare_history(stock_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(stock_data, pd.DataFrame) or stock_data.empty:
        raise ValueError("Stock history is required.")

    missing_columns = [
        column for column in RAW_FEATURE_COLUMNS if column not in stock_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Stock history is missing columns: {missing_columns}")

    history = stock_data[RAW_FEATURE_COLUMNS].copy()
    history.index = pd.to_datetime(history.index, errors="coerce")
    if history.index.isna().any():
        raise ValueError("Stock history contains an invalid date.")
    if history.index.duplicated().any():
        raise ValueError("Stock-history dates must be unique.")

    history = history.sort_index(kind="stable")
    for column in RAW_FEATURE_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    return history


def _calculate_rsi(prices: pd.Series) -> pd.Series:
    price_changes = prices.diff()
    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)
    average_gain = gains.rolling(14, min_periods=14).mean()
    average_loss = losses.rolling(14, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask(average_loss.eq(0) & average_gain.gt(0), 100.0)
    return rsi.mask(average_loss.eq(0) & average_gain.eq(0), 50.0)


def engineer_task7_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Task 6/7 features for one ticker without imputation."""
    features = _prepare_history(stock_data)
    adjusted_close = features["Adj Close"]

    features["Log_Return"] = np.log(adjusted_close / adjusted_close.shift(1))
    for lag_days in (1, 3, 5):
        features[f"Log_Return_Lag_{lag_days}"] = features["Log_Return"].shift(
            lag_days
        )

    features["Price_Range"] = features["High"] - features["Low"]
    previous_volume = features["Volume"].shift(1)
    features["Volume_Change"] = (
        features["Volume"] / previous_volume - 1
    ).replace([np.inf, -np.inf], np.nan)
    features["Volatility_20"] = features["Log_Return"].rolling(
        20, min_periods=20
    ).std()

    moving_average_10 = adjusted_close.rolling(10, min_periods=10).mean()
    moving_average_20 = adjusted_close.rolling(20, min_periods=20).mean()
    features["MA_10_MA_20_Ratio"] = (
        moving_average_10 / moving_average_20
    ).replace([np.inf, -np.inf], np.nan)
    features["Price_MA_20_Ratio"] = (
        adjusted_close / moving_average_20
    ).replace([np.inf, -np.inf], np.nan)
    features["RSI_14"] = _calculate_rsi(adjusted_close)

    macd = adjusted_close.ewm(span=12, adjust=False).mean() - adjusted_close.ewm(
        span=26, adjust=False
    ).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["MACD_Histogram"] = macd - macd_signal

    return features[MODEL_COLUMNS]


def calculate_return_metrics(
    actual_returns: Sequence[float], predicted_returns: Sequence[float]
) -> ReturnMetrics:
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("Actual and predicted returns must be aligned vectors.")
    if not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Return metrics require finite observations.")

    errors = actual - predicted
    residual_sum_of_squares = float(np.sum(errors**2))
    total_sum_of_squares = float(np.sum((actual - actual.mean()) ** 2))
    r2 = (
        1 - residual_sum_of_squares / total_sum_of_squares
        if total_sum_of_squares > 0
        else float("nan")
    )
    return ReturnMetrics(
        rmse=float(np.sqrt(np.mean(errors**2))),
        r2=float(r2),
        mae=float(np.mean(np.abs(errors))),
        directional_accuracy=float(
            np.mean(np.sign(actual) == np.sign(predicted)) * 100
        ),
    )


def reconstruct_adjusted_close(
    current_adjusted_close: float, predicted_log_return: float
) -> float:
    current_price = float(current_adjusted_close)
    predicted_return = float(predicted_log_return)
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError("Current adjusted close must be positive and finite.")
    if not math.isfinite(predicted_return):
        raise ValueError("Predicted log return must be finite.")
    return current_price * math.exp(predicted_return)


def _modelling_rows(stock_data: pd.DataFrame) -> pd.DataFrame:
    rows = engineer_task7_features(stock_data)
    adjusted_close = rows["Adj Close"]
    rows[TARGET_COLUMN] = np.log(adjusted_close.shift(-1) / adjusted_close)
    return rows.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).copy()


def evaluate_ticker_history(
    stock_data: pd.DataFrame,
    evaluation_model: ReturnModel,
    metadata: ModelMetadata,
) -> HistoricalEvaluation | None:
    """Evaluate only inside the artifact's immutable test-date window."""
    rows = _modelling_rows(stock_data)
    test_start = pd.Timestamp(metadata.test_start_date)
    test_end = pd.Timestamp(metadata.test_end_date)
    rows = rows.loc[(rows.index >= test_start) & (rows.index <= test_end)]
    if len(rows) < metadata.minimum_evaluation_rows:
        return None

    actual_returns = rows[TARGET_COLUMN].to_numpy(dtype=float)
    predicted_returns = np.asarray(
        evaluation_model.predict(rows[list(metadata.feature_columns)]),
        dtype=float,
    )
    if predicted_returns.shape != actual_returns.shape:
        raise ValueError("The evaluation model returned an unexpected shape.")

    return HistoricalEvaluation(
        model=calculate_return_metrics(actual_returns, predicted_returns),
        zero_return_baseline=calculate_return_metrics(
            actual_returns, np.zeros_like(actual_returns)
        ),
        observations=len(rows),
        first_evaluation_date=pd.Timestamp(rows.index.min()),
        last_evaluation_date=pd.Timestamp(rows.index.max()),
    )


def predict_next_return(
    stock_data: pd.DataFrame,
    production_model: ReturnModel,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> CurrentForecast:
    features = engineer_task7_features(stock_data)
    usable_rows = features.dropna(subset=list(feature_columns))
    if usable_rows.empty:
        raise ValueError("At least 21 valid trading rows are required for prediction.")

    latest_row = usable_rows.iloc[[-1]]
    predicted_values = np.asarray(
        production_model.predict(latest_row[list(feature_columns)]), dtype=float
    )
    if predicted_values.shape != (1,) or not np.isfinite(predicted_values).all():
        raise ValueError("The production model returned an invalid prediction.")

    current_adjusted_close = float(latest_row["Adj Close"].iloc[0])
    predicted_log_return = float(predicted_values[0])
    return CurrentForecast(
        as_of_date=pd.Timestamp(latest_row.index[0]),
        current_adjusted_close=current_adjusted_close,
        predicted_log_return=predicted_log_return,
        implied_next_adjusted_close=reconstruct_adjusted_close(
            current_adjusted_close, predicted_log_return
        ),
    )


def evaluation_scope(ticker: str, metadata: ModelMetadata) -> str:
    if ticker.strip().upper() in set(metadata.training_tickers):
        return "Seen-ticker temporal test"
    return "Unseen-ticker and future-date test"


def is_artifact_stale(
    metadata: ModelMetadata,
    *,
    today: date | None = None,
    stale_after_days: int = STALE_AFTER_DAYS,
) -> bool:
    current_date = today or date.today()
    data_end_date = pd.Timestamp(metadata.data_end_date).date()
    return (current_date - data_end_date).days > stale_after_days


def load_model_bundle(
    artifact_directory: Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> ModelBundle:
    """Load and validate the two versioned XGBoost model artifacts."""
    metadata_path = artifact_directory / "metadata.json"
    evaluation_path = artifact_directory / "xgb_next_log_return_evaluation.json"
    production_path = artifact_directory / "xgb_next_log_return_production.json"
    for path in (metadata_path, evaluation_path, production_path):
        if not path.is_file():
            raise FileNotFoundError(f"ML artifact is missing: {path}")

    metadata = ModelMetadata.from_dict(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    if metadata.schema_version != 1:
        raise ValueError("Unsupported ML artifact schema version.")
    if list(metadata.feature_columns) != FEATURE_COLUMNS:
        raise ValueError("ML artifact feature order does not match the app.")
    if metadata.target_column != TARGET_COLUMN:
        raise ValueError("ML artifact target does not match the app.")
    if not (
        pd.Timestamp(metadata.development_end_date)
        < pd.Timestamp(metadata.test_start_date)
        <= pd.Timestamp(metadata.test_end_date)
        <= pd.Timestamp(metadata.data_end_date)
    ):
        raise ValueError("ML artifact date boundaries are invalid.")

    import xgboost
    from xgboost import XGBRegressor

    if xgboost.__version__ != metadata.xgboost_version:
        raise ValueError(
            "ML artifact requires xgboost "
            f"{metadata.xgboost_version}, but {xgboost.__version__} is installed."
        )

    evaluation_model = XGBRegressor()
    evaluation_model.load_model(evaluation_path)
    production_model = XGBRegressor()
    production_model.load_model(production_path)
    return ModelBundle(evaluation_model, production_model, metadata)
