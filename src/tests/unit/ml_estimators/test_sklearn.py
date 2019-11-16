from unittest import TestCase
import os

from sklearn.linear_model import Ridge, Lasso
import pandas as pd
import numpy as np
from faker import Faker

from machine_learning.ml_estimators.sklearn import (
    AveragingRegressor,
    CorrelationSelector,
    EloRegressor,
)
from machine_learning.settings import BASE_DIR


FAKE = Faker()
ROW_COUNT = 10


class TestAveragingRegressor(TestCase):
    def setUp(self):
        data_frame = pd.DataFrame(
            {
                "year": ([2014] * round(ROW_COUNT * 0.2))
                + ([2015] * round(ROW_COUNT * 0.6))
                + ([2016] * round(ROW_COUNT * 0.2)),
                "prev_match_score": np.random.randint(50, 150, ROW_COUNT),
                "prev_match_oppo_score": np.random.randint(50, 150, ROW_COUNT),
                "round_number": 15,
                "margin": np.random.randint(5, 50, ROW_COUNT),
            }
        )

        self.X = data_frame.drop("margin", axis=1)
        self.y = data_frame["margin"]
        self.regressor = AveragingRegressor([("ridge", Ridge()), ("lasso", Lasso())])

    def test_predict(self):
        self.regressor.fit(self.X, self.y)
        predictions = self.regressor.predict(self.X)

        self.assertIsInstance(predictions, np.ndarray)


class TestCorrelationSelector(TestCase):
    def setUp(self):
        self.data_frame = pd.DataFrame(
            {
                "year": ([2014] * round(ROW_COUNT * 0.2))
                + ([2015] * round(ROW_COUNT * 0.6))
                + ([2016] * round(ROW_COUNT * 0.2)),
                "prev_match_score": np.random.randint(50, 150, ROW_COUNT),
                "prev_match_oppo_score": np.random.randint(50, 150, ROW_COUNT),
                "round_number": 15,
                "margin": np.random.randint(5, 50, ROW_COUNT),
            }
        )

        self.X = self.data_frame.drop("margin", axis=1)
        self.y = self.data_frame["margin"]
        self.selector = CorrelationSelector(labels=self.y)

    def test_transform(self):
        transformed_data_frame = self.selector.fit_transform(self.X)

        self.assertIsInstance(transformed_data_frame, pd.DataFrame)
        self.assertEqual(len(transformed_data_frame.columns), len(self.X.columns))

        with self.subTest("threshold > 0"):
            label_correlations = (
                self.data_frame.corr().fillna(0)["margin"].abs().sort_values()
            )
            threshold = label_correlations.iloc[round(len(label_correlations) * 0.5)]

            self.selector.threshold = threshold
            transformed_data_frame = self.selector.fit_transform(self.X)

            self.assertLess(len(transformed_data_frame.columns), len(self.X.columns))

        with self.subTest("cols_to_keep not empty"):
            cols_to_keep = [
                col for col in self.X.columns if col not in transformed_data_frame
            ][:2]

            self.selector.cols_to_keep = cols_to_keep
            transformed_data_frame = self.selector.fit_transform(self.X)

            for col in cols_to_keep:
                self.assertIn(col, transformed_data_frame.columns)


class TestEloRegressor(TestCase):
    def setUp(self):
        self.data_frame = (
            pd.read_json(os.path.join(BASE_DIR, "src/tests/fixtures/elo_data.json"))
            .set_index(["home_team", "year", "round_number"], drop=False)
            .rename_axis([None, None, None])
            .sort_index()
        )

        self.X = self.data_frame
        self.regressor = EloRegressor()

    def test_predict(self):
        X_train = self.X.query("year == 2014")
        X_test = self.X.query("year == 2015")
        # We don't use y when fitting the Elo model, so it can be just filler
        y = np.zeros(len(X_train))

        self.regressor.fit(X_train, y)
        predictions = self.regressor.predict(X_test)

        self.assertIsInstance(predictions, np.ndarray)

        with self.subTest("when there's a gap between match rounds"):
            invalid_X_test = X_test.query("round_number > 5")
            # Need to refit, because predict updates the state of the Elo ratings
            self.regressor.fit(X_train, y)

            with self.assertRaises(AssertionError):
                self.regressor.predict(invalid_X_test)
