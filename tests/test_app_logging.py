import io
import json
import logging
import unittest
from unittest.mock import patch

import app
import app_logging
import ml_prediction_ui as ml_ui


class StructuredLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        app_logging.start_request("unassigned")

    def test_log_event_contains_stable_context_and_request_id(self) -> None:
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(app_logging.JsonLogFormatter())
        logger = logging.getLogger("financial_dashboard.tests")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        request_id = app_logging.start_request("test-request")
        app_logging.log_event(
            logger,
            logging.WARNING,
            "yahoo_history_failed",
            ticker="AAPL",
            operation="history",
        )

        payload = json.loads(output.getvalue())
        self.assertEqual(request_id, "test-request")
        self.assertEqual(payload["event"], "yahoo_history_failed")
        self.assertEqual(payload["request_id"], "test-request")
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["operation"], "history")
        self.assertEqual(payload["level"], "WARNING")

    def test_configure_logging_does_not_add_duplicate_handlers(self) -> None:
        logger = app_logging.configure_logging()
        first_handler_count = len(logger.handlers)

        same_logger = app_logging.configure_logging()

        self.assertIs(same_logger, logger)
        self.assertEqual(len(same_logger.handlers), first_handler_count)

    def test_event_context_cannot_replace_the_correlation_id(self) -> None:
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(app_logging.JsonLogFormatter())
        logger = logging.getLogger("financial_dashboard.context-test")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        app_logging.start_request("trusted-request")

        app_logging.log_event(
            logger,
            logging.INFO,
            "context_checked",
            request_id="replacement-request",
        )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["request_id"], "trusted-request")

    def test_fundamentals_failure_is_logged_before_showing_fallback(self) -> None:
        with (
            patch.object(
                app,
                "get_fundamental_metrics",
                side_effect=RuntimeError("provider unavailable"),
            ),
            patch.object(app, "log_event") as log_event,
            patch.object(app.st, "subheader"),
            patch.object(app.st, "info"),
        ):
            app.render_fundamentals("AAPL")

        self.assertEqual(
            log_event.call_args.args[2],
            "fundamentals_render_failed",
        )
        self.assertEqual(log_event.call_args.kwargs["ticker"], "AAPL")
        self.assertTrue(log_event.call_args.kwargs["exc_info"])

    def test_ml_artifact_failure_is_logged_with_safe_context(self) -> None:
        with (
            patch.object(ml_ui.st, "button", return_value=True),
            patch.object(
                ml_ui,
                "get_instrument_quote_type",
                return_value="EQUITY",
            ),
            patch.object(
                ml_ui,
                "get_prediction_model_bundle",
                side_effect=ImportError("missing dependency"),
            ),
            patch.object(ml_ui, "log_event") as log_event,
            patch.object(ml_ui.st, "warning"),
        ):
            ml_ui.render_ml_prediction("MSFT")

        self.assertEqual(
            log_event.call_args.args[2],
            "ml_forecast_unavailable",
        )
        self.assertEqual(log_event.call_args.kwargs["ticker"], "MSFT")
        self.assertEqual(
            log_event.call_args.kwargs["error_type"],
            "ImportError",
        )


if __name__ == "__main__":
    unittest.main()
