import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

import ml_prediction


class ConstantReturnModel:
    def __init__(self, predicted_return: float) -> None:
        self.predicted_return = predicted_return
        self.last_columns: list[str] | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.last_columns = features.columns.tolist()
        return np.full(len(features), self.predicted_return, dtype=float)


def sample_history(number_of_rows: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=number_of_rows)
    trend = np.linspace(100.0, 125.0, number_of_rows)
    variation = np.sin(np.arange(number_of_rows) / 3.0)
    adjusted_close = trend + variation
    close = adjusted_close + 0.15
    return pd.DataFrame(
        {
            "Open": close - 0.40,
            "High": close + 1.10,
            "Low": close - 1.00,
            "Close": close,
            "Adj Close": adjusted_close,
            "Volume": 1_000_000 + np.arange(number_of_rows) * 2_500,
        },
        index=dates,
    )


class FeatureEngineeringTests(unittest.TestCase):
    def test_engineering_matches_task6_formulas_without_imputation(self) -> None:
        history = sample_history(40)

        features = ml_prediction.engineer_task7_features(history)

        adjusted_close = history["Adj Close"]
        expected_log_return = np.log(
            adjusted_close / adjusted_close.shift(1)
        )
        expected_ma10 = adjusted_close.rolling(10, min_periods=10).mean()
        expected_ma20 = adjusted_close.rolling(20, min_periods=20).mean()

        np.testing.assert_allclose(
            features["Log_Return"],
            expected_log_return,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            features["Log_Return_Lag_5"],
            expected_log_return.shift(5),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            features["MA_10_MA_20_Ratio"],
            expected_ma10 / expected_ma20,
            equal_nan=True,
        )
        self.assertTrue(features["Volatility_20"].iloc[:20].isna().all())
        self.assertEqual(features.columns.tolist(), ml_prediction.MODEL_COLUMNS)

    def test_duplicate_dates_are_rejected(self) -> None:
        history = sample_history(30)
        duplicated = pd.concat([history, history.iloc[[-1]]])

        with self.assertRaisesRegex(ValueError, "unique"):
            ml_prediction.engineer_task7_features(duplicated)


class ReturnMetricTests(unittest.TestCase):
    def test_metrics_are_calculated_in_return_space(self) -> None:
        actual = np.array([0.01, -0.02, 0.03, -0.01])
        predicted = np.array([0.00, -0.01, 0.02, -0.02])

        metrics = ml_prediction.calculate_return_metrics(actual, predicted)

        expected_rmse = math.sqrt(np.mean((actual - predicted) ** 2))
        expected_r2 = 1 - (
            np.sum((actual - predicted) ** 2)
            / np.sum((actual - actual.mean()) ** 2)
        )
        self.assertAlmostEqual(metrics.rmse, expected_rmse)
        self.assertAlmostEqual(metrics.r2, expected_r2)
        self.assertAlmostEqual(metrics.directional_accuracy, 75.0)

    def test_implied_price_is_reconstructed_separately(self) -> None:
        implied_price = ml_prediction.reconstruct_adjusted_close(100.0, 0.02)

        self.assertAlmostEqual(implied_price, 100.0 * math.exp(0.02))


class FrozenEvaluationTests(unittest.TestCase):
    def test_evaluation_never_expands_beyond_frozen_end_date(self) -> None:
        history = sample_history(90)
        engineered = ml_prediction.engineer_task7_features(history)
        usable_dates = engineered.dropna().index
        test_start = usable_dates[5].date().isoformat()
        test_end = usable_dates[14].date().isoformat()
        metadata = ml_prediction.ModelMetadata.for_tests(
            test_start_date=test_start,
            test_end_date=test_end,
            minimum_evaluation_rows=2,
        )
        model = ConstantReturnModel(0.0)

        evaluation = ml_prediction.evaluate_ticker_history(
            history,
            model,
            metadata,
        )

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertLessEqual(
            evaluation.last_evaluation_date,
            pd.Timestamp(test_end),
        )
        self.assertEqual(evaluation.observations, 10)
        self.assertEqual(model.last_columns, ml_prediction.FEATURE_COLUMNS)

    def test_insufficient_fixed_window_does_not_return_borrowed_metrics(
        self,
    ) -> None:
        history = sample_history(45)
        metadata = ml_prediction.ModelMetadata.for_tests(
            test_start_date="2025-02-20",
            test_end_date="2025-03-05",
            minimum_evaluation_rows=60,
        )

        evaluation = ml_prediction.evaluate_ticker_history(
            history,
            ConstantReturnModel(0.0),
            metadata,
        )

        self.assertIsNone(evaluation)

    def test_scope_distinguishes_seen_and_unseen_tickers(self) -> None:
        metadata = ml_prediction.ModelMetadata.for_tests(
            training_tickers=("MSFT", "NVDA"),
        )

        self.assertEqual(
            ml_prediction.evaluation_scope("MSFT", metadata),
            "Seen-ticker temporal test",
        )
        self.assertEqual(
            ml_prediction.evaluation_scope("AAPL", metadata),
            "Unseen-ticker and future-date test",
        )


class CurrentForecastTests(unittest.TestCase):
    def test_forecast_uses_latest_complete_feature_row(self) -> None:
        history = sample_history(80)
        model = ConstantReturnModel(0.01)

        forecast = ml_prediction.predict_next_return(history, model)

        self.assertEqual(forecast.as_of_date, history.index[-1])
        self.assertAlmostEqual(forecast.predicted_log_return, 0.01)
        self.assertAlmostEqual(
            forecast.implied_next_adjusted_close,
            history["Adj Close"].iloc[-1] * math.exp(0.01),
        )

    def test_artifact_staleness_uses_training_data_end_date(self) -> None:
        metadata = ml_prediction.ModelMetadata.for_tests(
            data_end_date="2026-06-30"
        )

        self.assertTrue(
            ml_prediction.is_artifact_stale(
                metadata,
                today=date(2026, 8, 14),
            )
        )


class PackagedArtifactTests(unittest.TestCase):
    def test_bundle_and_leave_one_out_evidence_share_one_contract(self) -> None:
        bundle = ml_prediction.load_model_bundle()
        unseen_evaluation = pd.read_csv(
            ml_prediction.DEFAULT_ARTIFACT_DIRECTORY
            / "unseen_ticker_evaluation.csv"
        )

        self.assertEqual(len(bundle.metadata.training_tickers), 18)
        self.assertEqual(len(unseen_evaluation), 18)
        self.assertEqual(
            set(unseen_evaluation["Held-out Ticker"]),
            set(bundle.metadata.training_tickers),
        )
        self.assertEqual(
            int(unseen_evaluation["Observations"].sum()),
            bundle.metadata.unseen_ticker_summary["observations"],
        )


if __name__ == "__main__":
    unittest.main()
