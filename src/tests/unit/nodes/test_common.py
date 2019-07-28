from unittest import TestCase


import pandas as pd

from tests.fixtures.data_factories import (
    fake_raw_match_results_data,
    fake_footywire_betting_data,
)
from machine_learning.nodes import common

START_DATE = "2012-01-01"
START_YEAR = int(START_DATE[:4])
END_DATE = "2013-12-31"
END_YEAR = int(END_DATE[:4]) + 1
N_MATCHES_PER_SEASON = 4
START_YEAR = 2013
END_YEAR = 2015
YEAR_RANGE = (2013, 2015)
REQUIRED_OUTPUT_COLS = ["home_team", "year", "round_number"]

# Need to multiply by two, because we add team & oppo_team row per match
N_TEAMMATCH_ROWS = N_MATCHES_PER_SEASON * len(range(*YEAR_RANGE)) * 2


class TestCommon(TestCase):
    def test_convert_to_data_frame(self):
        data = fake_raw_match_results_data(
            N_MATCHES_PER_SEASON, (START_YEAR, END_YEAR)
        ).to_dict("records")

        data_frames = common.convert_to_data_frame(data, data)

        self.assertEqual(len(data_frames), 2)

        for data_frame in data_frames:
            self.assertIsInstance(data_frame, pd.DataFrame)

        raw_data_fields = data[0].keys()
        data_frame_columns = data_frames[0].columns

        self.assertEqual(set(raw_data_fields), set(data_frame_columns))

    def test_combine_data(self):
        raw_betting_data = fake_footywire_betting_data(
            N_MATCHES_PER_SEASON, YEAR_RANGE, clean=False
        )
        min_year_range = min(YEAR_RANGE)
        older_data = fake_footywire_betting_data(
            N_MATCHES_PER_SEASON, (min_year_range - 2, min_year_range), clean=False
        ).append(raw_betting_data.query("season == @min_year_range"))

        combined_data = common.combine_data(raw_betting_data, older_data, axis=0)

        total_year_range = range(min_year_range - 2, max(YEAR_RANGE))
        self.assertEqual({*total_year_range}, {*combined_data["season"]})

        self.assertEqual(
            N_MATCHES_PER_SEASON * len(total_year_range), len(combined_data)
        )

        with self.subTest(axis=1):
            match_data = fake_raw_match_results_data(N_MATCHES_PER_SEASON, YEAR_RANGE)

            combined_data = common.combine_data(raw_betting_data, match_data, axis=1)

            self.assertEqual(
                N_MATCHES_PER_SEASON * len(range(*YEAR_RANGE)), len(combined_data)
            )
            self.assertEqual(
                len(raw_betting_data.columns) + len(match_data.columns),
                len(combined_data.columns),
            )
