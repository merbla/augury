"""Class for model trained on all AFL data and its associated data class."""

from typing import Optional, Union, Type

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

from augury.settings import TEAM_NAMES, ROUND_TYPES, VENUES, CATEGORY_COLS
from augury.sklearn.preprocessing import CorrelationSelector, ColumnDropper
from augury.types import R
from .base_ml_estimator import BaseMLEstimator, ELO_MODEL_COLS


SEED = 42
np.random.seed(SEED)

BEST_PARAMS = {
    "baggingregressor__base_estimator__booster": "dart",
    "baggingregressor__base_estimator__colsample_bylevel": 0.9593085973720467,
    "baggingregressor__base_estimator__colsample_bytree": 0.8366869579732328,
    "baggingregressor__base_estimator__learning_rate": 0.13118764001091077,
    "baggingregressor__base_estimator__max_depth": 6,
    "baggingregressor__base_estimator__n_estimators": 149,
    "baggingregressor__base_estimator__reg_alpha": 0.07296244459829336,
    "baggingregressor__base_estimator__reg_lambda": 0.11334834444556088,
    "baggingregressor__base_estimator__subsample": 0.8285733635843882,
    "baggingregressor__base_estimator__objective": "reg:squarederror",
    "baggingregressor__n_estimators": 7,
    "correlationselector__threshold": 0.030411689885916048,
}
PIPELINE = make_pipeline(
    ColumnDropper(cols_to_drop=ELO_MODEL_COLS),
    CorrelationSelector(cols_to_keep=CATEGORY_COLS),
    ColumnTransformer(
        [
            (
                "onehotencoder",
                OneHotEncoder(
                    categories=[TEAM_NAMES, TEAM_NAMES, ROUND_TYPES, VENUES],
                    sparse=False,
                    handle_unknown="ignore",
                ),
                CATEGORY_COLS,
            )
        ],
        remainder=StandardScaler(),
    ),
    BaggingRegressor(base_estimator=XGBRegressor(random_state=SEED, verbosity=0)),
).set_params(**BEST_PARAMS)


class BaggingEstimator(BaseMLEstimator):
    """Model for averaging predictions of an ensemble of models."""

    def __init__(
        self, pipeline: Pipeline = PIPELINE, name: Optional[str] = None
    ) -> None:
        """Instantiate a BaggingEstimator object.

        Params
        ------
        pipeline: Pipeline of Scikit-learn estimators ending in a regressor
            or classifier.
        name: Name of the estimator for reference by Kedro data sets and filenames.
        """
        super().__init__(pipeline=pipeline, name=name)

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Type[R]:
        """Fit estimator to the data."""
        assert (
            self.pipeline is not None
        ), "pipeline must be a scikit learn estimator but is None"

        pipeline_params = self.pipeline.get_params().keys()

        if "correlationselector__labels" in pipeline_params:
            self.pipeline.set_params(correlationselector__labels=y)

        return super().fit(X, y)
