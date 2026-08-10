import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app


class CloseFigureTests(unittest.TestCase):
    def test_close_figure_uses_usd_axis_and_no_persistent_markers(self) -> None:
        stock_data = pd.DataFrame(
            {"Close": [100.0, 101.5]},
            index=pd.to_datetime(["2026-08-06", "2026-08-07"]),
        )

        figure = app.build_close_figure(stock_data)

        self.assertEqual(figure.layout.yaxis.title.text, "Closing value (USD)")
        self.assertEqual(figure.data[0].mode, "lines")
        self.assertIn("USD", figure.data[0].hovertemplate)


class WatchlistStorageTests(unittest.TestCase):
    def test_watchlist_preserves_addition_order_and_ignores_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "watchlist.db"

            self.assertTrue(app.add_ticker_to_watchlist("aapl", database_path))
            self.assertTrue(app.add_ticker_to_watchlist("MSFT", database_path))
            self.assertFalse(app.add_ticker_to_watchlist(" AAPL ", database_path))

            self.assertEqual(
                app.load_watchlist(database_path),
                ["AAPL", "MSFT"],
            )

    def test_watchlist_rejects_invalid_ticker_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "watchlist.db"

            with self.assertRaises(ValueError):
                app.add_ticker_to_watchlist("../../etc/passwd", database_path)

    def test_watchlist_removes_only_the_selected_valid_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "watchlist.db"
            app.add_ticker_to_watchlist("AAPL", database_path)
            app.add_ticker_to_watchlist("MSFT", database_path)

            self.assertTrue(
                app.remove_ticker_from_watchlist("aapl", database_path)
            )
            self.assertFalse(
                app.remove_ticker_from_watchlist("AAPL", database_path)
            )
            self.assertEqual(app.load_watchlist(database_path), ["MSFT"])

    def test_watchlist_removal_rejects_invalid_ticker_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "watchlist.db"

            with self.assertRaises(ValueError):
                app.remove_ticker_from_watchlist(
                    "../../etc/passwd", database_path
                )


class WatchlistPriceTests(unittest.TestCase):
    def test_latest_close_and_one_day_percentage_change(self) -> None:
        closing_prices = pd.Series([100.0, 105.0])

        current_price, percentage_change = app.latest_daily_change(closing_prices)

        self.assertEqual(current_price, 105.0)
        self.assertAlmostEqual(percentage_change, 5.0)

    def test_latest_daily_change_uses_last_two_available_prices(self) -> None:
        closing_prices = pd.Series([100.0, None, 110.0])

        current_price, percentage_change = app.latest_daily_change(closing_prices)

        self.assertEqual(current_price, 110.0)
        self.assertAlmostEqual(percentage_change, 10.0)

    def test_watchlist_snapshot_returns_current_price_and_daily_change(self) -> None:
        market_data = pd.DataFrame(
            {"Close": [200.0, 204.0]},
            index=pd.to_datetime(["2026-08-07", "2026-08-10"]),
        )

        with patch.object(app.yf, "download", return_value=market_data):
            current_price, percentage_change = app.download_watchlist_snapshot(
                "AAPL"
            )

        self.assertEqual(current_price, 204.0)
        self.assertAlmostEqual(percentage_change, 2.0)


class WatchlistDisplayTests(unittest.TestCase):
    def test_watchlist_table_is_borderless_clickable_and_uses_shared_colours(
        self,
    ) -> None:
        table_html = app.build_watchlist_table(
            [
                ("AAPL", 229.35, 1.42),
                ("MSFT", 523.61, -0.68),
            ]
        )

        self.assertLess(table_html.index("AAPL"), table_html.index("MSFT"))
        self.assertIn("$229.35", table_html)
        self.assertIn("+1.42%", table_html)
        self.assertIn("-0.68%", table_html)
        self.assertIn("white-space: nowrap", table_html)
        self.assertIn('href="?ticker=AAPL"', table_html)
        self.assertIn('class="price positive"', table_html)
        self.assertIn('class="change positive"', table_html)
        self.assertIn('class="price negative"', table_html)
        self.assertIn("border: none !important", table_html)
        self.assertNotIn("border-bottom", table_html)
        self.assertNotIn("No.", table_html)
        self.assertNotIn('class="rank"', table_html)
        self.assertNotIn("Apple Inc", table_html)

    def test_watchlist_link_url_encodes_special_ticker_characters(self) -> None:
        table_html = app.build_watchlist_table([("^GSPC", 6400.0, 0.5)])

        self.assertIn('href="?ticker=%5EGSPC"', table_html)


class WatchlistNavigationTests(unittest.TestCase):
    def test_query_ticker_is_validated_and_normalized(self) -> None:
        self.assertEqual(
            app.selected_ticker_from_query({"ticker": " msft "}),
            "MSFT",
        )

    def test_invalid_query_ticker_is_ignored(self) -> None:
        self.assertEqual(
            app.selected_ticker_from_query({"ticker": "../../etc/passwd"}),
            "",
        )


class MarketSummaryTests(unittest.TestCase):
    def test_stock_download_uses_fixed_trailing_year(self) -> None:
        market_data = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-08-07"]),
        )

        with patch.object(
            app.yf, "download", return_value=market_data
        ) as download:
            returned_data = app.download_stock_data("AAPL")

        self.assertIs(returned_data, market_data)
        self.assertEqual(download.call_args.kwargs["period"], "1y")
        self.assertNotIn("start", download.call_args.kwargs)
        self.assertNotIn("end", download.call_args.kwargs)

    def test_latest_market_summary_uses_newest_available_row_and_close(self) -> None:
        market_data = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 108.0],
                "High": [106.0, 109.0, 112.0],
                "Low": [99.0, 104.0, 107.0],
                "Close": [104.0, None, 110.0],
            },
            index=pd.to_datetime(
                ["2026-08-06", "2026-08-07", "2026-08-10"]
            ),
        )

        summary = app.latest_market_summary(market_data)

        self.assertEqual(summary["date"], pd.Timestamp("2026-08-10"))
        self.assertEqual(summary["open"], 108.0)
        self.assertEqual(summary["high"], 112.0)
        self.assertEqual(summary["low"], 107.0)
        self.assertAlmostEqual(summary["return"], (110.0 / 104.0 - 1) * 100)

    def test_fundamental_metrics_include_only_available_values(self) -> None:
        metrics = app.format_fundamental_metrics(
            {
                "trailingPE": 31.234,
                "dividendRate": 1.04,
                "dividendYield": 0.42,
                "lastDividendValue": 0.26,
                "trailingEps": 7.18,
                "exDividendDate": 1786665600,
                "marketCap": 3_420_000_000_000,
                "sharesOutstanding": 15_200_000_000,
                "unused": None,
            }
        )

        self.assertEqual(
            metrics,
            [
                ("P/E Ratio", "31.23"),
                ("Dividend", "$1.04 · 0.42%"),
                ("Quarterly Dividend", "$0.26"),
                ("EPS", "$7.18"),
                ("Ex-Dividend Date", "2026-08-14"),
                ("Market Cap", "$3.42T"),
                ("Shares Outstanding", "15.20B"),
            ],
        )

    def test_fundamental_metrics_omit_missing_and_invalid_values(self) -> None:
        metrics = app.format_fundamental_metrics(
            {
                "trailingPE": None,
                "dividendYield": float("nan"),
                "exDividendDate": "not-a-date",
                "marketCap": 2_500_000,
            }
        )

        self.assertEqual(metrics, [("Market Cap", "$2.50M")])


if __name__ == "__main__":
    unittest.main()
