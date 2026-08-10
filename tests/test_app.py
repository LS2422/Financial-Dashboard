import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
