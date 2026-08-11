import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app


class CloseFigureTests(unittest.TestCase):
    def test_positive_return_uses_green_line_without_persistent_markers(
        self,
    ) -> None:
        stock_data = pd.DataFrame(
            {"Close": [100.0, 101.5]},
            index=pd.to_datetime(["2026-08-06", "2026-08-07"]),
        )

        figure = app.build_close_figure(stock_data)

        self.assertEqual(figure.layout.yaxis.title.text, "Closing value (USD)")
        self.assertEqual(figure.data[0].mode, "lines")
        self.assertEqual(figure.data[0].line.color, "#22c55e")
        self.assertIn("USD", figure.data[0].hovertemplate)

    def test_negative_return_uses_red_line(self) -> None:
        stock_data = pd.DataFrame(
            {"Close": [101.5, 100.0]},
            index=pd.to_datetime(["2026-08-06", "2026-08-07"]),
        )

        figure = app.build_close_figure(stock_data)

        self.assertEqual(figure.data[0].line.color, "#ef6461")

    def test_explicit_latest_return_colours_a_single_visible_point(self) -> None:
        stock_data = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-08-07"]),
        )

        figure = app.build_close_figure(stock_data, momentum_return=-1.25)

        self.assertEqual(figure.data[0].line.color, "#ef6461")


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
    def test_stock_download_includes_indicator_lookback_and_no_fixed_end(
        self,
    ) -> None:
        market_data = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-08-07"]),
        )

        with patch.object(
            app.yf, "download", return_value=market_data
        ) as download:
            returned_data = app.download_stock_data("AAPL", "2026-01-01")

        self.assertIs(returned_data, market_data)
        self.assertEqual(download.call_args.kwargs["start"], "2024-11-27")
        self.assertNotIn("period", download.call_args.kwargs)
        self.assertNotIn("end", download.call_args.kwargs)

    def test_visible_data_starts_on_selected_date(self) -> None:
        market_data = pd.DataFrame(
            {"Close": [98.0, 100.0, 102.0]},
            index=pd.to_datetime(
                ["2025-12-31", "2026-01-02", "2026-01-05"]
            ),
        )

        visible_data = app.filter_data_from_start(
            market_data,
            "2026-01-01",
        )

        self.assertEqual(
            list(visible_data.index),
            list(pd.to_datetime(["2026-01-02", "2026-01-05"])),
        )

    def test_technical_indicators_use_full_hidden_history(self) -> None:
        closing_prices = pd.Series(range(1, 221), dtype=float)
        market_data = pd.DataFrame(
            {"Close": closing_prices.to_numpy()},
            index=pd.date_range("2025-01-01", periods=220, freq="D"),
        )

        enriched_data = app.add_technical_indicators(market_data)

        self.assertAlmostEqual(enriched_data["MA_50"].iloc[-1], 195.5)
        self.assertAlmostEqual(enriched_data["MA_200"].iloc[-1], 120.5)
        self.assertAlmostEqual(enriched_data["RSI_14"].iloc[-1], 100.0)
        self.assertTrue(pd.isna(enriched_data["MA_200"].iloc[198]))

    def test_rsi_is_neutral_when_closing_price_is_flat(self) -> None:
        market_data = pd.DataFrame(
            {"Close": [100.0] * 20},
            index=pd.date_range("2026-01-01", periods=20, freq="D"),
        )

        enriched_data = app.add_technical_indicators(market_data)

        self.assertEqual(enriched_data["RSI_14"].iloc[-1], 50.0)

    def test_latest_market_summary_uses_newest_available_row_and_close(self) -> None:
        market_data = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 108.0],
                "High": [106.0, 109.0, 112.0],
                "Low": [99.0, 104.0, 107.0],
                "Close": [104.0, None, 110.0],
                "Volume": [1_000_000, 1_100_000, 1_250_000],
                "MA_50": [100.0, 101.0, 102.0],
                "MA_200": [90.0, 91.0, 92.0],
                "RSI_14": [55.0, 56.0, 57.0],
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
        self.assertEqual(summary["volume"], 1_250_000.0)
        self.assertEqual(summary["ma_50"], 102.0)
        self.assertEqual(summary["ma_200"], 92.0)
        self.assertEqual(summary["rsi"], 57.0)
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
                ("Dividend", "$1.04"),
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

    def test_fundamentals_fall_back_when_detailed_info_fails(self) -> None:
        class FallbackTicker:
            def get_info(self):
                raise RuntimeError("Detailed metadata is unavailable")

            def get_fast_info(self):
                return {
                    "lastPrice": 100.0,
                    "marketCap": 2_500_000_000,
                    "shares": 25_000_000,
                }

            def get_income_stmt(self, **_kwargs):
                return pd.DataFrame(
                    {pd.Timestamp("2026-06-30"): [5.0]},
                    index=["DilutedEPS"],
                )

            def get_dividends(self, **_kwargs):
                return pd.Series([0.25, 0.25, 0.25, 0.25])

            def get_calendar(self):
                return {"Ex-Dividend Date": pd.Timestamp("2026-08-10")}

        with patch.object(app.yf, "Ticker", return_value=FallbackTicker()):
            metrics = app.get_fundamental_metrics("FALLBACK")

        self.assertEqual(
            metrics,
            [
                ("P/E Ratio", "20.00"),
                ("Dividend", "$1.00"),
                ("EPS", "$5.00"),
                ("Ex-Dividend Date", "2026-08-10"),
                ("Market Cap", "$2.50B"),
                ("Shares Outstanding", "25.00M"),
            ],
        )

    def test_fundamentals_show_a_visible_unavailable_state(self) -> None:
        with (
            patch.object(app, "get_fundamental_metrics", return_value=[]),
            patch.object(app.st, "subheader") as subheader,
            patch.object(app.st, "info") as info,
        ):
            app.render_fundamentals("AAPL")

        subheader.assert_called_once_with("Company Fundamentals")
        info.assert_called_once_with(
            "Company fundamentals are temporarily unavailable from Yahoo "
            "Finance."
        )

    def test_metric_grid_uses_equal_label_and_value_sizes_and_two_columns(
        self,
    ) -> None:
        metric_html = app.build_metric_grid(
            [
                ("P/E Ratio", "31.23", "neutral"),
                ("Return", "-1.08%", "negative"),
            ],
            layout="fundamentals",
            accessible_name="Company fundamentals",
        )

        self.assertIn("metric-grid--fundamentals", metric_html)
        self.assertIn(
            "grid-template-columns: repeat(2", app.METRIC_GRID_STYLES
        )
        self.assertIn("font-size: 1rem", app.METRIC_GRID_STYLES)
        self.assertIn('class="metric-value negative"', metric_html)
        self.assertNotIn("font-size: 2", app.METRIC_GRID_STYLES)

    def test_latest_market_grid_uses_four_columns_for_two_balanced_rows(
        self,
    ) -> None:
        self.assertIn(
            ".metric-grid--latest {\n    grid-template-columns: repeat(4",
            app.METRIC_GRID_STYLES,
        )

    def test_latest_market_grid_displays_moving_averages_and_rsi(self) -> None:
        market_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1_000, 1_100],
                "MA_50": [95.0, 96.0],
                "MA_200": [85.0, 86.0],
                "RSI_14": [55.0, 56.0],
            },
            index=pd.to_datetime(["2026-08-07", "2026-08-10"]),
        )

        with (
            patch.object(app, "build_metric_grid", return_value="<grid>") as grid,
            patch.object(app.st, "subheader"),
            patch.object(app.st, "caption"),
            patch.object(app.st, "markdown"),
        ):
            app.render_market_summary(market_data)

        metrics = grid.call_args.args[0]
        self.assertEqual(
            [label for label, _, _ in metrics],
            [
                "Open",
                "High",
                "Low",
                "Return",
                "Volume",
                "Moving Average 50D",
                "Moving Average 200D",
                "RSI (14D)",
            ],
        )
        self.assertIn(("Moving Average 50D", "$96.00", "neutral"), metrics)
        self.assertIn(("Moving Average 200D", "$86.00", "neutral"), metrics)
        self.assertIn(("RSI (14D)", "56.00", "neutral"), metrics)


class LatestNewsTests(unittest.TestCase):
    def test_news_parser_returns_three_newest_relevant_safe_stories(self) -> None:
        raw_news = [
            {
                "title": "Older Apple story",
                "publisher": "Publisher A",
                "link": "https://finance.yahoo.com/news/older-apple",
                "providerPublishTime": 1_786_440_000,
                "relatedTickers": ["AAPL"],
            },
            {
                "title": "Newest unrelated story",
                "publisher": "Publisher B",
                "link": "https://finance.yahoo.com/news/unrelated",
                "providerPublishTime": 1_786_460_000,
                "relatedTickers": ["MSFT"],
            },
            {
                "title": "Unsafe Apple link",
                "publisher": "Publisher C",
                "link": "javascript:alert(1)",
                "providerPublishTime": 1_786_455_000,
                "relatedTickers": ["AAPL"],
            },
            {
                "title": "Malformed Apple link",
                "publisher": "Publisher C",
                "link": "https://finance.yahoo.com:invalid/news/apple",
                "providerPublishTime": 1_786_454_000,
                "relatedTickers": ["AAPL"],
            },
            {
                "title": "Newest Apple story",
                "publisher": "Publisher D",
                "link": "https://finance.yahoo.com/news/newest-apple",
                "providerPublishTime": 1_786_450_000,
                "relatedTickers": ["AAPL"],
            },
            {
                "title": "Second Apple story",
                "publisher": "Publisher E",
                "link": "https://finance.yahoo.com/news/second-apple",
                "providerPublishTime": 1_786_445_000,
                "relatedTickers": ["AAPL"],
            },
            {
                "title": "Fourth Apple story",
                "publisher": "Publisher F",
                "link": "https://finance.yahoo.com/news/fourth-apple",
                "providerPublishTime": 1_786_430_000,
                "relatedTickers": ["AAPL"],
            },
        ]

        news_items = app.normalize_news_items(raw_news, "AAPL", limit=3)

        self.assertEqual(
            [item["title"] for item in news_items],
            [
                "Newest Apple story",
                "Second Apple story",
                "Older Apple story",
            ],
        )
        self.assertTrue(
            all(item["url"].startswith("https://finance.yahoo.com/") for item in news_items)
        )

    def test_nested_news_schema_and_html_are_safely_rendered(self) -> None:
        raw_news = [
            {
                "content": {
                    "title": '<script>alert("news")</script> Apple update',
                    "provider": {"displayName": "Yahoo & Partners"},
                    "pubDate": "2026-08-11T14:30:39Z",
                    "clickThroughUrl": {
                        "url": "https://finance.yahoo.com/news/apple-update"
                    },
                }
            }
        ]

        news_items = app.normalize_news_items(raw_news, "AAPL")
        news_html = app.build_news_list(news_items)

        self.assertEqual(len(news_items), 1)
        self.assertIn("2026-08-11 14:30 UTC", news_html)
        self.assertIn("Yahoo &amp; Partners", news_html)
        self.assertIn("&lt;script&gt;", news_html)
        self.assertNotIn("<script>", news_html)
        self.assertIn('rel="noopener noreferrer"', news_html)

    def test_news_section_shows_an_explicit_empty_state(self) -> None:
        with (
            patch.object(app, "get_latest_news", return_value=[]),
            patch.object(app.st, "subheader") as subheader,
            patch.object(app.st, "info") as info,
        ):
            app.render_latest_news("AAPL")

        subheader.assert_called_once_with("Latest News")
        info.assert_called_once_with(
            "No recent Yahoo Finance news is available for this symbol."
        )


if __name__ == "__main__":
    unittest.main()
