"""Module for data cleaning functions"""

from typing import Optional, Pattern, Callable, List
import re

import pandas as pd

from machine_learning.nodes import match
from machine_learning.nodes.base import _parse_dates
from machine_learning.data_config import TEAM_TRANSLATIONS


MATCH_COL_TRANSLATIONS = {
    "home_points": "home_score",
    "away_points": "away_score",
    "margin": "home_margin",
    "season": "year",
    "game": "match_id",
    "home_goals": "home_team_goals",
    "away_goals": "away_team_goals",
    "home_behinds": "home_team_behinds",
    "away_behinds": "away_team_behinds",
}
PLAYER_COL_TRANSLATIONS = {
    "time_on_ground__": "time_on_ground",
    "id": "player_id",
    "round": "round_number",
    "season": "year",
}
REGULAR_ROUND: Pattern = re.compile(r"round\s+(\d+)$", flags=re.I)

UNUSED_PLAYER_COLS = [
    "local_start_time",
    "attendance",
    "hq1g",
    "hq1b",
    "hq2g",
    "hq2b",
    "hq3g",
    "hq3b",
    "hq4g",
    "hq4b",
    "aq1g",
    "aq1b",
    "aq2g",
    "aq2b",
    "aq3g",
    "aq3b",
    "aq4g",
    "aq4b",
    "jumper_no_",
    "umpire_1",
    "umpire_2",
    "umpire_3",
    "umpire_4",
    "substitute",
    "group_id",
]

PLAYER_FILLNA = {
    "first_name": "",
    "surname": "",
    "player_id": 0,
    "playing_for": "",
    "kicks": 0,
    "marks": 0,
    "handballs": 0,
    "goals": 0,
    "behinds": 0,
    "hit_outs": 0,
    "tackles": 0,
    "rebounds": 0,
    "inside_50s": 0,
    "clearances": 0,
    "clangers": 0,
    "frees_for": 0,
    "frees_against": 0,
    "brownlow_votes": 0,
    "contested_possessions": 0,
    "uncontested_possessions": 0,
    "contested_marks": 0,
    "marks_inside_50": 0,
    "one_percenters": 0,
    "bounces": 0,
    "goal_assists": 0,
    "time_on_ground": 0,
    "player_name": "",
}

LABEL_COLS = ["score", "oppo_score"]


def _translate_team_name(team_name: str) -> str:
    return TEAM_TRANSLATIONS[team_name] if team_name in TEAM_TRANSLATIONS else team_name


def _translate_team_column(col_name: str) -> Callable[[pd.DataFrame], str]:
    return lambda data_frame: data_frame[col_name].map(_translate_team_name)


# ID values are converted to floats automatically, making for awkward strings later.
# We want them as strings, because sometimes we have to use player names as replacement
# IDs, and we concatenate multiple ID values to create a unique index.
def _convert_id_to_string(id_label: str) -> Callable:
    return lambda df: df[id_label].astype(int).astype(str)


def _player_id_col(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Need to add year to ID, because there are some
    # player_id/match_id combos, decades apart, that by chance overlap
    return (
        data_frame["year"].astype(str)
        + "."
        + data_frame["match_id"].astype(str)
        + "."
        + data_frame["player_id"].astype(str)
    )


def _clean_roster_data(
    player_data_frame: pd.DataFrame, roster_data: pd.DataFrame
) -> pd.DataFrame:
    if not roster_data.any().any():
        return roster_data.assign(player_id=[])

    roster_data_frame = (
        roster_data.assign(date=_parse_dates)
        .rename(columns={"season": "year"})
        .merge(
            player_data_frame[["player_name", "player_id"]],
            on=["player_name"],
            how="left",
        )
        .sort_values("player_id", ascending=False)
        # There are some duplicate player names over the years, so we drop the oldest,
        # hoping that the contemporary player matches the one with the most-recent
        # entry into the AFL. If two players with the same name are playing in the
        # league at the same time, that will likely result in errors
        .drop_duplicates(subset=["player_name"], keep="first")
    )

    # If a player is new to the league, he won't have a player_id per AFL Tables data,
    # so we make one up just using his name
    roster_data_frame["player_id"].fillna(
        roster_data_frame["player_name"], inplace=True
    )

    return roster_data_frame.assign(id=_player_id_col).set_index("id")


def _append_rosters_to_player_data(
    player_data: pd.DataFrame, roster_data: pd.DataFrame
) -> pd.DataFrame:
    if roster_data is None:
        return player_data

    cleaned_roster_data_frame = _clean_roster_data(player_data, roster_data)

    return pd.concat([player_data, cleaned_roster_data_frame], sort=False).fillna(0)


def _filter_out_dodgy_data(duplicate_subset=None) -> Callable:
    return lambda df: (
        df.sort_values("date", ascending=True)
        # Some early matches (1800s) have fully-duplicated rows.
        # Also, drawn finals get replayed, which screws up my indexing and a bunch of other
        # data munging, so getting match_ids for the repeat matches, and filtering
        # them out of the data frame
        .drop_duplicates(subset=duplicate_subset, keep="last")
        # There were some weird round-robin rounds in the early days, and it's easier to
        # drop them rather than figure out how to split up the rounds.
        .query(
            "(year != 1897 | round_number != 15) "
            "& (year != 1924 | round_number != 19)"
        )
    )


def clean_player_data(
    player_data: pd.DataFrame,
    match_data: pd.DataFrame,
    roster_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    cleaned_player_data = (
        player_data.rename(columns=PLAYER_COL_TRANSLATIONS)
        .astype({"year": int})
        .assign(
            # Some player data venues have trailing spaces
            venue=lambda x: x["venue"].str.strip(),
            player_name=lambda x: x["first_name"] + " " + x["surname"],
            player_id=_convert_id_to_string("player_id"),
            home_team=_translate_team_column("home_team"),
            away_team=_translate_team_column("away_team"),
            playing_for=_translate_team_column("playing_for"),
            date=_parse_dates,
        )
        .drop(UNUSED_PLAYER_COLS + ["first_name", "surname", "round_number"], axis=1)
        # Player data match IDs are wrong for recent years.
        # The easiest way to add correct ones is to graft on the IDs
        # from match_results. Also, match_results round_numbers are integers rather than
        # a mix of ints and strings.
        .merge(
            match_data.pipe(match.clean_match_data).loc[
                :, ["date", "venue", "round_number", "match_id"]
            ],
            on=["date", "venue"],
            how="left",
        )
        .pipe(
            _filter_out_dodgy_data(
                duplicate_subset=["year", "round_number", "player_id"]
            )
        )
        .drop("venue", axis=1)
        # brownlow_votes aren't known until the end of the season
        .fillna({"brownlow_votes": 0})
        # Joining on date/venue leaves two duplicates played at M.C.G.
        # on 29-4-1986 & 9-8-1986, but that's an acceptable loss of data
        # and easier than munging team names
        .dropna()
        # Need to add year to ID, because there are some
        # player_id/match_id combos, decades apart, that by chance overlap
        .assign(id=_player_id_col)
        .set_index("id")
        .sort_index()
    )

    return _append_rosters_to_player_data(cleaned_player_data, roster_data)


def clean_joined_data(data_frames: List[pd.DataFrame]):
    # We need to sort by length (going from longest to shortest), then keeping first
    # duplicated column to make sure we don't lose earlier values of shared columns
    # (e.g. dropping match data's 'date' column in favor of the betting data 'date'
    # column results in lots of NaT values, because betting data only goes back to 2010)
    sorted_data_frames = sorted(data_frames, key=len, reverse=True)
    joined_data_frame = pd.concat(sorted_data_frames, axis=1)
    duplicate_columns = joined_data_frame.columns.duplicated(keep="first")

    return joined_data_frame.loc[:, ~duplicate_columns]
