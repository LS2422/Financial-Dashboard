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
    def test_watchlist_table_keeps_compact_ranked_ticker_rows(self) -> None:
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
        self.assertNotIn("Apple Inc", table_html)


if __name__ == "__main__":
    unittest.main()
