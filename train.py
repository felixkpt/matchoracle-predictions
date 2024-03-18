import sys
from configs.active_competitions.competitions_data import get_competition_ids, get_trained_competitions
from app.train_predictions.hda_predictions import hda_predictions
from app.train_predictions.bts_predictions import bts_predictions
from app.train_predictions.over25_predictions import over25_predictions
from app.train_predictions.cs_predictions import cs_predictions
from app.matches.load_matches import load_for_training
from configs.logger import Logger
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Define constants
HISTORY_LIMITS = [5, 7, 10, 15, 20, 25]
TRAIN_TO_DATE = datetime.strptime('2023-10-31', '%Y-%m-%d')
from_date = TRAIN_TO_DATE - relativedelta(months=48)
to_date = TRAIN_TO_DATE - relativedelta(days=1)


def train(user_token, extra_args):
    print("\n............... START TRAIN PREDICTIONS ..................\n")

    train_to_date = TRAIN_TO_DATE.strftime("%Y-%m-%d")
    competition_ids = []

    # Extract competition IDs from extra_args
    for arg in extra_args:
        if arg.startswith('--competition='):
            competition_ids = [int(arg.split('=')[1])]

    # Fetch competition IDs from the backend API if not provided
    if not competition_ids:
        competition_ids = get_competition_ids(user_token)

    trained_competition_ids = get_trained_competitions()

    for history_limit_per_match in HISTORY_LIMITS:
        PREDICTION_TYPE = f"regular_prediction_last_{history_limit_per_match}_matches_optimized_30"

        for COMPETITION_ID in competition_ids:
            if not trained_competition_ids or COMPETITION_ID not in trained_competition_ids:
                Logger.info(f"Competition: {COMPETITION_ID}")
                Logger.info(f"Updating training to date: {train_to_date}\n")
                Logger.info(f"Prediction type: {PREDICTION_TYPE}\n")

                be_params = from_date, to_date, history_limit_per_match
                ignore_saved = any(arg.startswith('ignore-saved')
                                   for arg in extra_args)

                compe_data = {'id': COMPETITION_ID,
                              'prediction_type': PREDICTION_TYPE}

                # Load train and test data for all targets
                train_matches, test_matches = load_for_training(
                    compe_data, user_token, be_params, per_page=20, train_ratio=.75, ignore_saved=ignore_saved)

                total_matches = len(train_matches) + len(test_matches)

                # Calculate the percentages
                train_percentage = (
                    int(round((len(train_matches) / total_matches) * 100)) if total_matches > 0 else 0)
                test_percentage = (
                    int(round((len(test_matches) / total_matches) * 100)) if total_matches > 0 else 0)

                Logger.info(
                    f"Number of train matches: {len(train_matches)}, ({train_percentage})%")
                Logger.info(
                    f"Number of test matches: {len(test_matches)}, ({test_percentage})%")

                if total_matches == 0:
                    print('No matches to make predictions!')
                    return

                is_grid_search = any(arg.startswith('grid-search')
                                     for arg in extra_args)
                is_random_search = False
                update_model = True

                hda_predictions(user_token, train_matches, test_matches, compe_data,
                                PREDICTION_TYPE, is_grid_search, is_random_search=is_random_search, update_model=update_model)

                bts_predictions(user_token, train_matches, test_matches, compe_data,
                                PREDICTION_TYPE, is_grid_search, is_random_search=is_random_search, update_model=update_model)

                over25_predictions(user_token, train_matches, test_matches, compe_data,
                                   PREDICTION_TYPE, is_grid_search, is_random_search=is_random_search, update_model=update_model)

                cs_predictions(user_token, train_matches, test_matches, compe_data,
                               PREDICTION_TYPE, is_grid_search, is_random_search=is_random_search, update_model=update_model)

    print(f"\n....... END TRAIN PREDICTIONS, Happy coding! ........")
