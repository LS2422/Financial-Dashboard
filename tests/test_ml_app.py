import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

import ml_prediction
import ml_prediction_ui as ml_ui


def sample_history() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=80)
    return pd.DataFrame(
        {
            "Open": range(100, 180),
            "High": range(102, 182),
            "Low": range(99, 179),
            "Close": range(101, 181),
            "Adj Close": range(101, 181),
            "Volume": range(1_000_000, 1_000_080),
        },
        index=dates,
    )


class PredictionDataTests(unittest.TestCase):
    def setUp(self) -> None:
        ml_ui.download_prediction_history.clear()

    def test_prediction_history_uses_artifact_start_and_latest_available_end(
        self,
    ) -> None:
        history = sample_history()
        with patch.object(
            ml_ui.MARKET_DATA_SOURCE,
            "history",
            return_value=history,
        ) as download:
            result = ml_ui.download_prediction_history("MSFT", "2020-12-01")

        self.assertEqual(len(result), len(history))
        download.assert_called_once_with(
            "MSFT",
            start_date="2020-12-01",
        )

    def test_prediction_history_rejects_missing_required_columns(self) -> None:
        incomplete_history = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-08-12"]),
        )
        with patch.object(
            ml_ui.MARKET_DATA_SOURCE,
            "history",
            return_value=incomplete_history,
        ):
            with self.assertRaisesRegex(ValueError, "required price data"):
                ml_ui.download_prediction_history("BROKEN", "2020-12-01")


class InstrumentEligibilityTests(unittest.TestCase):
    def test_only_yahoo_equity_quote_type_is_eligible(self) -> None:
        self.assertTrue(ml_ui.is_equity_quote_type("EQUITY"))
        self.assertFalse(ml_ui.is_equity_quote_type("INDEX"))
        self.assertFalse(ml_ui.is_equity_quote_type("FUTURE"))
        self.assertFalse(ml_ui.is_equity_quote_type(None))


class PredictionPresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = ml_prediction.ModelBundle(
            evaluation_model=MagicMock(),
            production_model=MagicMock(),
            metadata=ml_prediction.ModelMetadata.for_tests(
                test_start_date="2025-02-03",
                test_end_date="2025-03-31",
                data_end_date="2026-06-01",
                training_tickers=("MSFT", "NVDA"),
                minimum_evaluation_rows=2,
            ),
        )
        self.forecast = ml_prediction.CurrentForecast(
            as_of_date=pd.Timestamp("2026-08-12"),
            current_adjusted_close=100.0,
            predicted_log_return=0.0123,
            implied_next_adjusted_close=101.2376,
        )
        self.evaluation = ml_prediction.HistoricalEvaluation(
            model=ml_prediction.ReturnMetrics(
                rmse=0.0200,
                r2=-0.12,
                mae=0.015,
                directional_accuracy=48.0,
            ),
            zero_return_baseline=ml_prediction.ReturnMetrics(
                rmse=0.0180,
                r2=-0.01,
                mae=0.014,
                directional_accuracy=0.0,
            ),
            observations=40,
            first_evaluation_date=pd.Timestamp("2025-02-03"),
            last_evaluation_date=pd.Timestamp("2025-03-31"),
        )

    def test_button_is_lazy_and_does_not_load_artifacts_before_click(self) -> None:
        with (
            patch.object(ml_ui.st, "button", return_value=False),
            patch.object(ml_ui, "get_prediction_model_bundle") as load_bundle,
            patch.object(ml_ui, "download_prediction_history") as download,
        ):
            ml_ui.render_ml_prediction("MSFT")

        load_bundle.assert_not_called()
        download.assert_not_called()

    def test_seen_ticker_displays_return_metrics_and_fixed_scope(self) -> None:
        with (
            patch.object(ml_ui.st, "button", return_value=True),
            patch.object(ml_ui, "get_instrument_quote_type", return_value="EQUITY"),
            patch.object(ml_ui, "get_prediction_model_bundle", return_value=self.bundle),
            patch.object(ml_ui, "download_prediction_history", return_value=sample_history()),
            patch.object(ml_ui.ml_prediction, "predict_next_return", return_value=self.forecast),
            patch.object(ml_ui.ml_prediction, "evaluate_ticker_history", return_value=self.evaluation),
            patch.object(ml_ui.ml_prediction, "is_artifact_stale", return_value=False),
            patch.object(ml_ui.st, "columns", return_value=[MagicMock() for _ in range(4)]),
            patch.object(ml_ui.st, "metric") as metric,
            patch.object(ml_ui.st, "caption") as caption,
            patch.object(ml_ui.st, "warning") as warning,
        ):
            ml_ui.render_ml_prediction("MSFT", today=date(2026, 8, 14))

        labels_and_values = [call.args[:2] for call in metric.call_args_list]
        self.assertIn(("Predicted next log return", "+1.230%"), labels_and_values)
        self.assertIn(("Implied next adjusted close", "$101.24"), labels_and_values)
        self.assertIn(("Log-return RMSE", "2.000 pp"), labels_and_values)
        self.assertIn(("Log-return R²", "-0.120"), labels_and_values)
        self.assertTrue(
            any(
                "Seen-ticker temporal test" in str(call.args[0])
                and "2025-02-03 to 2025-03-31" in str(call.args[0])
                for call in caption.call_args_list
            )
        )
        warning_messages = [str(call.args[0]) for call in warning.call_args_list]
        self.assertTrue(any("zero-return baseline" in message for message in warning_messages))
        self.assertTrue(any("negative" in message for message in warning_messages))

    def test_unseen_ticker_without_fixed_history_keeps_forecast_only(self) -> None:
        unseen_summary = {
            "pooled_rmse": 0.0185,
            "pooled_r2": -0.01,
            "observations": 5_076,
            "held_out_tickers": 18,
        }
        unseen_metadata = ml_prediction.ModelMetadata.for_tests(
            training_tickers=("MSFT", "NVDA"),
            minimum_evaluation_rows=60,
        )
        unseen_metadata = ml_prediction.ModelMetadata(
            **{
                **unseen_metadata.__dict__,
                "unseen_ticker_summary": unseen_summary,
            }
        )
        bundle = ml_prediction.ModelBundle(
            evaluation_model=MagicMock(),
            production_model=MagicMock(),
            metadata=unseen_metadata,
        )

        with (
            patch.object(ml_ui.st, "button", return_value=True),
            patch.object(ml_ui, "get_instrument_quote_type", return_value="EQUITY"),
            patch.object(ml_ui, "get_prediction_model_bundle", return_value=bundle),
            patch.object(ml_ui, "download_prediction_history", return_value=sample_history()),
            patch.object(ml_ui.ml_prediction, "predict_next_return", return_value=self.forecast),
            patch.object(ml_ui.ml_prediction, "evaluate_ticker_history", return_value=None),
            patch.object(ml_ui.ml_prediction, "is_artifact_stale", return_value=False),
            patch.object(ml_ui.st, "columns", return_value=[MagicMock() for _ in range(4)]),
            patch.object(ml_ui.st, "metric") as metric,
            patch.object(ml_ui.st, "caption") as caption,
            patch.object(ml_ui.st, "warning") as warning,
            patch.object(ml_ui.st, "info") as info,
        ):
            ml_ui.render_ml_prediction("AAPL", today=date(2026, 8, 14))

        metric_labels = [call.args[0] for call in metric.call_args_list]
        self.assertIn("Predicted next log return", metric_labels)
        self.assertNotIn("Log-return RMSE", metric_labels)
        self.assertTrue(
            any("outside the 18-stock" in str(call.args[0]) for call in warning.call_args_list)
        )
        self.assertTrue(
            any(
                "ticker-specific rmse" in str(call.args[0]).lower()
                for call in info.call_args_list
            )
        )
        self.assertTrue(
            any("18 held-out tickers" in str(call.args[0]) for call in caption.call_args_list)
        )

    def test_non_equity_symbol_is_rejected_before_loading_model(self) -> None:
        with (
            patch.object(ml_ui.st, "button", return_value=True),
            patch.object(ml_ui, "get_instrument_quote_type", return_value="INDEX"),
            patch.object(ml_ui, "get_prediction_model_bundle") as load_bundle,
            patch.object(ml_ui.st, "info") as info,
        ):
            ml_ui.render_ml_prediction("^GSPC")

        load_bundle.assert_not_called()
        self.assertIn("available only for stocks", info.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
