import os

EMAIL = "admin@example.com"
PASSWORD = "admin@example.com"

API_BASE_URL = "http://matchoracle-be.local/api"

HISTORY_LIMITS = [7, 10, 12, 15]

# Define predictors used for training
COMMON_FEATURES = [
    'hour',
    'day_of_week',
    'competition_id',
    'season_id',
    'country_id',
    'referees_ids',
    'home_team_id',
    'away_team_id',

    'home_team_totals',
    'home_team_wins',
    'home_team_draws',
    'home_team_loses',
    'home_team_goals_for',
    'home_team_goals_for_avg',
    'home_team_goals_against',
    'home_team_goals_against_avg',
    'home_team_bts_games',
    'home_team_over15_games',
    'home_team_over25_games',
    'home_team_over35_games',

    'away_team_totals',
    'away_team_wins',
    'away_team_draws',
    'away_team_loses',
    'away_team_goals_for',
    'away_team_goals_for_avg',
    'away_team_goals_against',
    'away_team_goals_against_avg',
    'home_team_bts_games',
    'away_team_over15_games',
    'away_team_over25_games',
    'away_team_over35_games',

    'ht_home_team_totals',
    'ht_home_team_wins',
    'ht_home_team_draws',
    'ht_home_team_loses',
    'ht_home_team_goals_for',
    'ht_home_team_goals_for_avg',
    'ht_home_team_goals_against',
    'ht_home_team_goals_against_avg',
    'ht_home_team_bts_games',
    'ht_home_team_over15_games',
    'ht_home_team_over25_games',
    'ht_home_team_over35_games',

    'ht_away_team_totals',
    'ht_away_team_wins',
    'ht_away_team_draws',
    'ht_away_team_loses',
    'ht_away_team_goals_for',
    'ht_away_team_goals_for_avg',
    'ht_away_team_goals_against',
    'ht_away_team_goals_against_avg',
    'ht_home_team_bts_games',
    'ht_away_team_over15_games',
    'ht_away_team_over25_games',
    'ht_away_team_over35_games',

    'h2h_home_team_totals',
    'h2h_home_team_wins',
    'h2h_home_team_draws',
    'h2h_home_team_loses',
    'h2h_home_team_goals_for',
    'h2h_home_team_goals_for_avg',
    'h2h_home_team_goals_against',
    'h2h_home_team_goals_against_avg',
    'h2h_home_team_bts_games',
    'h2h_home_team_over15_games',
    'h2h_home_team_over25_games',
    'h2h_home_team_over35_games',

    'h2h_away_team_totals',
    'h2h_away_team_wins',
    'h2h_away_team_draws',
    'h2h_away_team_loses',
    'h2h_away_team_goals_for',
    'h2h_away_team_goals_for_avg',
    'h2h_away_team_goals_against',
    'h2h_away_team_goals_against_avg',
    'h2h_away_team_bts_games',
    'h2h_away_team_over15_games',
    'h2h_away_team_over25_games',
    'h2h_away_team_over35_games',

    'h2h_ht_home_team_totals',
    'h2h_ht_home_team_wins',
    'h2h_ht_home_team_draws',
    'h2h_ht_home_team_loses',
    'h2h_ht_home_team_goals_for',
    'h2h_ht_home_team_goals_for_avg',
    'h2h_ht_home_team_goals_against',
    'h2h_ht_home_team_goals_against_avg',
    'h2h_ht_home_team_bts_games',
    'h2h_ht_home_team_over15_games',
    'h2h_ht_home_team_over25_games',
    'h2h_ht_home_team_over35_games',

    'h2h_ht_away_team_totals',
    'h2h_ht_away_team_wins',
    'h2h_ht_away_team_draws',
    'h2h_ht_away_team_loses',
    'h2h_ht_away_team_goals_for',
    'h2h_ht_away_team_goals_for_avg',
    'h2h_ht_away_team_goals_against',
    'h2h_ht_away_team_goals_against_avg',
    'h2h_ht_away_team_bts_games',
    'h2h_ht_away_team_over15_games',
    'h2h_ht_away_team_over25_games',
    'h2h_ht_away_team_over35_games',
]

GRID_SEARCH_N_SPLITS = 3
GRID_SEARCH_VARBOSE = 0
TRAIN_VARBOSE = 0


def basepath():
    # Get the part before "configs"
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_before_configs, configs_directory = os.path.split(current_directory)
    return path_before_configs
