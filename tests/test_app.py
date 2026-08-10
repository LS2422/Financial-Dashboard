import unittest

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


if __name__ == "__main__":
    unittest.main()
