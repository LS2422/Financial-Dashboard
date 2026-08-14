import unittest
from unittest.mock import MagicMock

import pandas as pd

import market_data


class MarketDataInterfaceTests(unittest.TestCase):
    def test_an_alternative_data_source_can_satisfy_the_protocol(self) -> None:
        class ExampleMarketDataSource:
            def history(self, ticker, *, start_date=None, period=None):
                return pd.DataFrame()

            def quote_type(self, ticker):
                return "EQUITY"

            def market_details(self, ticker):
                return market_data.MarketDetails("Example", "NASDAQ")

            def fundamentals(self, ticker):
                return {}

            def news(self, ticker, *, count=12):
                return []

        self.assertIsInstance(
            ExampleMarketDataSource(),
            market_data.MarketDataSource,
        )

    def test_yahoo_history_normalizes_ticker_and_preserves_request_options(
        self,
    ) -> None:
        yahoo = MagicMock()
        history = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-08-12"]),
        )
        yahoo.download.return_value = history
        source = market_data.YahooFinanceDataSource(yahoo)

        result = source.history(" aapl ", start_date="2026-01-01")

        self.assertIs(result, history)
        yahoo.download.assert_called_once_with(
            "AAPL",
            start="2026-01-01",
            interval="1d",
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )

    def test_yahoo_history_requires_exactly_one_time_range(self) -> None:
        source = market_data.YahooFinanceDataSource(MagicMock())

        with self.assertRaisesRegex(ValueError, "start_date or period"):
            source.history("AAPL")
        with self.assertRaisesRegex(ValueError, "start_date or period"):
            source.history(
                "AAPL",
                start_date="2026-01-01",
                period="5d",
            )

    def test_yahoo_history_rejects_an_unexpected_response_type(self) -> None:
        yahoo = MagicMock()
        yahoo.download.return_value = "not a dataframe"
        source = market_data.YahooFinanceDataSource(yahoo)

        with self.assertRaisesRegex(ValueError, "dataframe"):
            source.history("AAPL", period="5d")

    def test_quote_type_uses_ticker_metadata_when_search_is_unavailable(
        self,
    ) -> None:
        yahoo = MagicMock()
        yahoo.Search.side_effect = RuntimeError("search unavailable")
        yahoo.Ticker.return_value.get_info.return_value = {
            "quoteType": "equity"
        }
        source = market_data.YahooFinanceDataSource(yahoo)

        self.assertEqual(source.quote_type("msft"), "EQUITY")

    def test_fundamentals_keep_independent_yahoo_fallbacks(self) -> None:
        class FallbackTicker:
            def get_info(self):
                raise RuntimeError("detailed info unavailable")

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

        yahoo = MagicMock()
        yahoo.Ticker.return_value = FallbackTicker()
        source = market_data.YahooFinanceDataSource(yahoo)

        fundamentals = source.fundamentals("fallback")

        self.assertEqual(fundamentals["trailingPE"], 20.0)
        self.assertEqual(fundamentals["dividendRate"], 1.0)
        self.assertEqual(fundamentals["trailingEps"], 5.0)
        self.assertEqual(fundamentals["marketCap"], 2_500_000_000)
        self.assertEqual(fundamentals["sharesOutstanding"], 25_000_000)
        self.assertEqual(
            fundamentals["exDividendDate"],
            pd.Timestamp("2026-08-10"),
        )

    def test_news_returns_only_mapping_records(self) -> None:
        yahoo = MagicMock()
        yahoo.Search.return_value.news = [
            {"title": "Valid story"},
            "invalid story",
            None,
        ]
        source = market_data.YahooFinanceDataSource(yahoo)

        self.assertEqual(
            source.news("AAPL", count=3),
            [{"title": "Valid story"}],
        )


if __name__ == "__main__":
    unittest.main()
