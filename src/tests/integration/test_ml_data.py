from unittest import TestCase, skipIf
import os
import pandas as pd

from tests.helpers import KedroContextMixin
from augury.ml_data import MLData


class TestMLData(TestCase, KedroContextMixin):
    def setUp(self):
        context = self.load_context(start_date="2014-01-01", end_date="2015-12-31",)
        self.ml_data = MLData(
            context=context,
            # We don't use any data set, but this makes sure we don't overwrite
            # one that actually matters
            data_set="fake_data",
            train_year_range=(2014, 2015),
            test_year_range=(2015, 2016),
            update_data=True,
            # This stops just short of the step that writes to a JSON file
            to_nodes=["final_model_data"],
        )

    @skipIf(
        os.getenv("CI", "").lower() == "true",
        "More trouble than it's worth trying to get usable data sets in CI."
        "Also, given my difficulties getting player data loaded in CI, running this "
        "is likely equally impossible.",
    )
    # full includes betting, match, and player, so no reason to test them separately
    def test_full_pipeline(self):
        self.ml_data.pipeline = "full"

        self.assertIsInstance(self.ml_data.data, pd.DataFrame)

    @skipIf(
        os.getenv("CI", "").lower() == "true",
        "More trouble than it's worth trying to get usable data sets in CI."
        "Also, given my difficulties getting player data loaded in CI, running this "
        "is likely equally impossible.",
    )
    def test_legacy_pipeline(self):
        self.ml_data.pipeline = "legacy"

        self.assertIsInstance(self.ml_data.data, pd.DataFrame)
