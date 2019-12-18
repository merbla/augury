"""Pipeline construction."""

from kedro.pipeline import Pipeline, node

from machine_learning.settings import CATEGORY_COLS
from machine_learning.nodes import common, feature_calculation
from .player_pipeline import create_player_pipeline
from .betting_pipeline import create_betting_pipeline
from .match_pipeline import create_match_pipeline


DEFAULT_FEATURE_CALCS = [
    (feature_calculation.calculate_multiplication, [("win_odds", "ladder_position")])
]


def create_full_pipeline(
    start_date: str,
    end_date: str,
    match_data_set="final_match_data",
    feature_calcs=DEFAULT_FEATURE_CALCS,
    final_data_set="model_data",
    category_cols=CATEGORY_COLS + ["prev_match_oppo_team"],
):
    return Pipeline(
        [
            node(common.convert_to_data_frame, "final_betting_data", "betting_df"),
            node(common.convert_to_data_frame, match_data_set, "match_df"),
            node(common.convert_to_data_frame, "final_player_data", "player_df"),
            node(
                common.combine_data(axis=1),
                ["betting_data_df", "match_data_df", "player_data_df"],
                "joined_data",
            ),
            node(
                common.filter_by_date(start_date, end_date),
                "joined_data",
                "filtered_data",
            ),
            node(
                feature_calculation.feature_calculator(feature_calcs),
                "filtered_data",
                "data_a",
            ),
            node(
                common.sort_data_frame_columns(category_cols=category_cols),
                "data_a",
                "data_b",
            ),
            node(common.finalize_data, "data_b", "data_c", name="final_model_data"),
            node(common.convert_to_json, "data_c", final_data_set),
        ]
    )
