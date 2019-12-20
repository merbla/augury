from unittest import TestCase

from kedro.context import load_context

from augury.settings import BASE_DIR
from augury.ml_estimators import ConfidenceEstimator


class TestConfidenceEstimator(TestCase):
    def setUp(self):
        context = load_context(BASE_DIR)
        self.loaded_model = context.catalog.load(ConfidenceEstimator().name)

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, ConfidenceEstimator)
