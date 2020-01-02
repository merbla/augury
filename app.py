"""Bottle server routes for mimicking serverless HTTP API.

See the serverless entrypoint main.py for documentation.
"""

import os
import sys
from datetime import date

from bottle import Bottle, run, request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from augury import api


TRUE = "true"
FALSE = "false"

app = Bottle()


@app.route("/predictions")
def predictions():
    """Generate predictions for the given year and round number."""
    this_year = date.today().year
    year_range_param = request.query.year_range or f"{this_year}-{this_year + 1}"
    year_range = tuple([int(year) for year in year_range_param.split("-")])

    round_number = request.query.round_number
    round_number = int(round_number) if round_number is not None else None

    ml_models_param = request.query.ml_models
    ml_models_param = (
        ml_models_param.split(",")
        if ml_models_param is not None and ml_models_param != ""
        else None
    )

    return api.make_predictions(
        year_range, round_number=round_number, ml_model_names=ml_models_param
    )


@app.route("/fixtures")
def fixtures():
    """Fetch fixture data for the given date range."""
    start_date = request.query.start_date
    end_date = request.query.end_date

    return api.fetch_fixture_data(start_date, end_date)


@app.route("/match_results")
def match_results():
    """Fetch match results data for the given date range."""
    start_date = request.query.start_date
    end_date = request.query.end_date

    return api.fetch_match_results_data(start_date, end_date)


@app.route("/ml_models")
def ml_models():
    """Fetch info for all available ML models."""
    return api.fetch_ml_model_info()


run(app, host="0.0.0.0", port=8008, reloader=True)
