"""Module for data transformations and internal conventions"""

# TODO: The data refactor reordered the columns, which completely messed up
# predictions. I don't want to retrain the models, so I'll just use the original column
# list to make sure they're in the same order as before, and figure out a better
# solution later
ORIGINAL_COLUMNS = [
    "team",
    "oppo_team",
    "round_type",
    "venue",
    "win_odds",
    "line_odds",
    "oppo_win_odds",
    "oppo_line_odds",
    "betting_pred_win",
    "rolling_betting_pred_win_rate",
    "oppo_result",
    "oppo_betting_pred_win",
    "oppo_rolling_betting_pred_win_rate",
    "rolling_prev_match_kicks_sum",
    "rolling_prev_match_kicks_max",
    "rolling_prev_match_kicks_min",
    "rolling_prev_match_kicks_skew",
    "rolling_prev_match_kicks_std",
    "rolling_prev_match_marks_sum",
    "rolling_prev_match_marks_max",
    "rolling_prev_match_marks_min",
    "rolling_prev_match_marks_skew",
    "rolling_prev_match_marks_std",
    "rolling_prev_match_handballs_sum",
    "rolling_prev_match_handballs_max",
    "rolling_prev_match_handballs_min",
    "rolling_prev_match_handballs_skew",
    "rolling_prev_match_handballs_std",
    "rolling_prev_match_goals_sum",
    "rolling_prev_match_goals_max",
    "rolling_prev_match_goals_min",
    "rolling_prev_match_goals_skew",
    "rolling_prev_match_goals_std",
    "rolling_prev_match_behinds_sum",
    "rolling_prev_match_behinds_max",
    "rolling_prev_match_behinds_min",
    "rolling_prev_match_behinds_skew",
    "rolling_prev_match_behinds_std",
    "rolling_prev_match_hit_outs_sum",
    "rolling_prev_match_hit_outs_max",
    "rolling_prev_match_hit_outs_min",
    "rolling_prev_match_hit_outs_skew",
    "rolling_prev_match_hit_outs_std",
    "rolling_prev_match_tackles_sum",
    "rolling_prev_match_tackles_max",
    "rolling_prev_match_tackles_min",
    "rolling_prev_match_tackles_skew",
    "rolling_prev_match_tackles_std",
    "rolling_prev_match_rebounds_sum",
    "rolling_prev_match_rebounds_max",
    "rolling_prev_match_rebounds_min",
    "rolling_prev_match_rebounds_skew",
    "rolling_prev_match_rebounds_std",
    "rolling_prev_match_inside_50s_sum",
    "rolling_prev_match_inside_50s_max",
    "rolling_prev_match_inside_50s_min",
    "rolling_prev_match_inside_50s_skew",
    "rolling_prev_match_inside_50s_std",
    "rolling_prev_match_clearances_sum",
    "rolling_prev_match_clearances_max",
    "rolling_prev_match_clearances_min",
    "rolling_prev_match_clearances_skew",
    "rolling_prev_match_clearances_std",
    "rolling_prev_match_clangers_sum",
    "rolling_prev_match_clangers_max",
    "rolling_prev_match_clangers_min",
    "rolling_prev_match_clangers_skew",
    "rolling_prev_match_clangers_std",
    "rolling_prev_match_frees_for_sum",
    "rolling_prev_match_frees_for_max",
    "rolling_prev_match_frees_for_min",
    "rolling_prev_match_frees_for_skew",
    "rolling_prev_match_frees_for_std",
    "rolling_prev_match_frees_against_sum",
    "rolling_prev_match_frees_against_max",
    "rolling_prev_match_frees_against_min",
    "rolling_prev_match_frees_against_skew",
    "rolling_prev_match_frees_against_std",
    "rolling_prev_match_contested_possessions_sum",
    "rolling_prev_match_contested_possessions_max",
    "rolling_prev_match_contested_possessions_min",
    "rolling_prev_match_contested_possessions_skew",
    "rolling_prev_match_contested_possessions_std",
    "rolling_prev_match_uncontested_possessions_sum",
    "rolling_prev_match_uncontested_possessions_max",
    "rolling_prev_match_uncontested_possessions_min",
    "rolling_prev_match_uncontested_possessions_skew",
    "rolling_prev_match_uncontested_possessions_std",
    "rolling_prev_match_contested_marks_sum",
    "rolling_prev_match_contested_marks_max",
    "rolling_prev_match_contested_marks_min",
    "rolling_prev_match_contested_marks_skew",
    "rolling_prev_match_contested_marks_std",
    "rolling_prev_match_marks_inside_50_sum",
    "rolling_prev_match_marks_inside_50_max",
    "rolling_prev_match_marks_inside_50_min",
    "rolling_prev_match_marks_inside_50_skew",
    "rolling_prev_match_marks_inside_50_std",
    "rolling_prev_match_one_percenters_sum",
    "rolling_prev_match_one_percenters_max",
    "rolling_prev_match_one_percenters_min",
    "rolling_prev_match_one_percenters_skew",
    "rolling_prev_match_one_percenters_std",
    "rolling_prev_match_bounces_sum",
    "rolling_prev_match_bounces_max",
    "rolling_prev_match_bounces_min",
    "rolling_prev_match_bounces_skew",
    "rolling_prev_match_bounces_std",
    "rolling_prev_match_goal_assists_sum",
    "rolling_prev_match_goal_assists_max",
    "rolling_prev_match_goal_assists_min",
    "rolling_prev_match_goal_assists_skew",
    "rolling_prev_match_goal_assists_std",
    "rolling_prev_match_time_on_ground_sum",
    "rolling_prev_match_time_on_ground_max",
    "rolling_prev_match_time_on_ground_min",
    "rolling_prev_match_time_on_ground_skew",
    "rolling_prev_match_time_on_ground_std",
    "last_year_brownlow_votes_sum",
    "last_year_brownlow_votes_max",
    "last_year_brownlow_votes_min",
    "last_year_brownlow_votes_skew",
    "last_year_brownlow_votes_std",
    "oppo_rolling_prev_match_kicks_sum",
    "oppo_rolling_prev_match_kicks_max",
    "oppo_rolling_prev_match_kicks_min",
    "oppo_rolling_prev_match_kicks_skew",
    "oppo_rolling_prev_match_kicks_std",
    "oppo_rolling_prev_match_marks_sum",
    "oppo_rolling_prev_match_marks_max",
    "oppo_rolling_prev_match_marks_min",
    "oppo_rolling_prev_match_marks_skew",
    "oppo_rolling_prev_match_marks_std",
    "oppo_rolling_prev_match_handballs_sum",
    "oppo_rolling_prev_match_handballs_max",
    "oppo_rolling_prev_match_handballs_min",
    "oppo_rolling_prev_match_handballs_skew",
    "oppo_rolling_prev_match_handballs_std",
    "oppo_rolling_prev_match_goals_sum",
    "oppo_rolling_prev_match_goals_max",
    "oppo_rolling_prev_match_goals_min",
    "oppo_rolling_prev_match_goals_skew",
    "oppo_rolling_prev_match_goals_std",
    "oppo_rolling_prev_match_behinds_sum",
    "oppo_rolling_prev_match_behinds_max",
    "oppo_rolling_prev_match_behinds_min",
    "oppo_rolling_prev_match_behinds_skew",
    "oppo_rolling_prev_match_behinds_std",
    "oppo_rolling_prev_match_hit_outs_sum",
    "oppo_rolling_prev_match_hit_outs_max",
    "oppo_rolling_prev_match_hit_outs_min",
    "oppo_rolling_prev_match_hit_outs_skew",
    "oppo_rolling_prev_match_hit_outs_std",
    "oppo_rolling_prev_match_tackles_sum",
    "oppo_rolling_prev_match_tackles_max",
    "oppo_rolling_prev_match_tackles_min",
    "oppo_rolling_prev_match_tackles_skew",
    "oppo_rolling_prev_match_tackles_std",
    "oppo_rolling_prev_match_rebounds_sum",
    "oppo_rolling_prev_match_rebounds_max",
    "oppo_rolling_prev_match_rebounds_min",
    "oppo_rolling_prev_match_rebounds_skew",
    "oppo_rolling_prev_match_rebounds_std",
    "oppo_rolling_prev_match_inside_50s_sum",
    "oppo_rolling_prev_match_inside_50s_max",
    "oppo_rolling_prev_match_inside_50s_min",
    "oppo_rolling_prev_match_inside_50s_skew",
    "oppo_rolling_prev_match_inside_50s_std",
    "oppo_rolling_prev_match_clearances_sum",
    "oppo_rolling_prev_match_clearances_max",
    "oppo_rolling_prev_match_clearances_min",
    "oppo_rolling_prev_match_clearances_skew",
    "oppo_rolling_prev_match_clearances_std",
    "oppo_rolling_prev_match_clangers_sum",
    "oppo_rolling_prev_match_clangers_max",
    "oppo_rolling_prev_match_clangers_min",
    "oppo_rolling_prev_match_clangers_skew",
    "oppo_rolling_prev_match_clangers_std",
    "oppo_rolling_prev_match_frees_for_sum",
    "oppo_rolling_prev_match_frees_for_max",
    "oppo_rolling_prev_match_frees_for_min",
    "oppo_rolling_prev_match_frees_for_skew",
    "oppo_rolling_prev_match_frees_for_std",
    "oppo_rolling_prev_match_frees_against_sum",
    "oppo_rolling_prev_match_frees_against_max",
    "oppo_rolling_prev_match_frees_against_min",
    "oppo_rolling_prev_match_frees_against_skew",
    "oppo_rolling_prev_match_frees_against_std",
    "oppo_rolling_prev_match_contested_possessions_sum",
    "oppo_rolling_prev_match_contested_possessions_max",
    "oppo_rolling_prev_match_contested_possessions_min",
    "oppo_rolling_prev_match_contested_possessions_skew",
    "oppo_rolling_prev_match_contested_possessions_std",
    "oppo_rolling_prev_match_uncontested_possessions_sum",
    "oppo_rolling_prev_match_uncontested_possessions_max",
    "oppo_rolling_prev_match_uncontested_possessions_min",
    "oppo_rolling_prev_match_uncontested_possessions_skew",
    "oppo_rolling_prev_match_uncontested_possessions_std",
    "oppo_rolling_prev_match_contested_marks_sum",
    "oppo_rolling_prev_match_contested_marks_max",
    "oppo_rolling_prev_match_contested_marks_min",
    "oppo_rolling_prev_match_contested_marks_skew",
    "oppo_rolling_prev_match_contested_marks_std",
    "oppo_rolling_prev_match_marks_inside_50_sum",
    "oppo_rolling_prev_match_marks_inside_50_max",
    "oppo_rolling_prev_match_marks_inside_50_min",
    "oppo_rolling_prev_match_marks_inside_50_skew",
    "oppo_rolling_prev_match_marks_inside_50_std",
    "oppo_rolling_prev_match_one_percenters_sum",
    "oppo_rolling_prev_match_one_percenters_max",
    "oppo_rolling_prev_match_one_percenters_min",
    "oppo_rolling_prev_match_one_percenters_skew",
    "oppo_rolling_prev_match_one_percenters_std",
    "oppo_rolling_prev_match_bounces_sum",
    "oppo_rolling_prev_match_bounces_max",
    "oppo_rolling_prev_match_bounces_min",
    "oppo_rolling_prev_match_bounces_skew",
    "oppo_rolling_prev_match_bounces_std",
    "oppo_rolling_prev_match_goal_assists_sum",
    "oppo_rolling_prev_match_goal_assists_max",
    "oppo_rolling_prev_match_goal_assists_min",
    "oppo_rolling_prev_match_goal_assists_skew",
    "oppo_rolling_prev_match_goal_assists_std",
    "oppo_rolling_prev_match_time_on_ground_sum",
    "oppo_rolling_prev_match_time_on_ground_max",
    "oppo_rolling_prev_match_time_on_ground_min",
    "oppo_rolling_prev_match_time_on_ground_skew",
    "oppo_rolling_prev_match_time_on_ground_std",
    "oppo_last_year_brownlow_votes_sum",
    "oppo_last_year_brownlow_votes_max",
    "oppo_last_year_brownlow_votes_min",
    "oppo_last_year_brownlow_votes_skew",
    "oppo_last_year_brownlow_votes_std",
    "goals",
    "behinds",
    "score",
    "oppo_goals",
    "oppo_behinds",
    "oppo_score",
    "year",
    "round_number",
    "at_home",
    "out_of_state",
    "travel_distance",
    "result",
    "margin",
    "prev_match_score",
    "prev_match_oppo_score",
    "prev_match_result",
    "prev_match_margin",
    "prev_match_goals",
    "prev_match_behinds",
    "cum_win_points",
    "win_streak",
    "elo_rating",
    "rolling_prev_match_result_rate",
    "rolling_mean_margin_by_oppo_team",
    "rolling_mean_result_by_oppo_team",
    "rolling_mean_score_by_oppo_team",
    "rolling_mean_margin_by_venue",
    "rolling_mean_result_by_venue",
    "rolling_mean_score_by_venue",
    "oppo_travel_distance",
    "oppo_prev_match_score",
    "oppo_prev_match_oppo_score",
    "oppo_prev_match_result",
    "oppo_prev_match_margin",
    "oppo_prev_match_goals",
    "oppo_prev_match_behinds",
    "oppo_cum_win_points",
    "oppo_win_streak",
    "oppo_elo_rating",
    "oppo_rolling_prev_match_result_rate",
    "oppo_rolling_mean_margin_by_oppo_team",
    "oppo_rolling_mean_result_by_oppo_team",
    "oppo_rolling_mean_score_by_oppo_team",
    "oppo_rolling_mean_margin_by_venue",
    "oppo_rolling_mean_result_by_venue",
    "oppo_rolling_mean_score_by_venue",
    "cum_percent",
    "ladder_position",
    "elo_pred_win",
    "rolling_elo_pred_win_rate",
    "elo_rating_divided_by_ladder_position",
    "oppo_cum_percent",
    "oppo_ladder_position",
    "elo_rating_divided_by_win_odds",
    "win_odds_multiplied_by_ladder_position",
]
