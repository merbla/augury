from unittest import TestCase

from kedro.context import load_context

from augury.settings import BASE_DIR
from augury.ml_estimators import ConfidenceEstimator


class TestConfidenceEstimator(TestCase):
    def setUp(self):
        # Need to use production environment for loading model if in CI, because we
        # don't check model files into source control
        kedro_env = (
            "production"
            if os.environ.get("CI") == "true"
            else os.environ.get("PYTHON_ENV")
        )
        context = load_context(BASE_DIR, env=kedro_env)
        self.loaded_model = context.catalog.load(ConfidenceEstimator().name)

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, ConfidenceEstimator)
