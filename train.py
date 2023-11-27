from app.train_predictions.hda_predictions import hda_predictions
from app.train_predictions.bts_predictions import bts_predictions
from app.train_predictions.over25_predictions import over25_predictions
from app.train_predictions.cs_predictions import cs_predictions
from app.train_predictions.cs_predictions_normalizer import cs_predictions_normalizer
from app.matches.load_matches import load_for_training
from configs.logger import Logger
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys

# Define constants
HISTORY_LIMITS = [5, 7, 10, 15, 20, 25]
# HISTORY_LIMITS = [7]
# Campeonato Brasileiro Série A, Championship, EPL, Portugal primera, LaLiga
# 47, 48, 125, 148
COMPETITION_IDS = [25, 47, 48, 125, 148]
# COMPETITION_IDS = [125, 148]

# Calculate from_date and to_date
TRAIN_TO_DATE = datetime.strptime('2023-09-30', '%Y-%m-%d')
from_date = TRAIN_TO_DATE - relativedelta(months=48)
to_date = TRAIN_TO_DATE - relativedelta(days=1)


def train(user_token):
    print("\n............... START TRAIN PREDICTIONS ..................\n")

    train_to_date = TRAIN_TO_DATE.strftime("%Y-%m-%d")

    for history_limit_per_match in HISTORY_LIMITS:

        PREDICTION_TYPE = f"regular_prediction_last_{history_limit_per_match}_matches_optimized_30"

        for COMPETITION_ID in COMPETITION_IDS:
            compe_data = {}
            compe_data['id'] = COMPETITION_ID
            compe_data['prediction_type'] = PREDICTION_TYPE

            Logger.info(f"Competition: {COMPETITION_ID}")
            Logger.info(f"Updating training to date: {train_to_date}\n")
            Logger.info(f"Prediction type: {PREDICTION_TYPE}\n")

            is_grid_search = False
            ignore_saved = False
            for arg in sys.argv:
                if arg.startswith('grid-search'):
                    is_grid_search = True
                if arg.startswith('ignore-saved'):
                    ignore_saved = True

            be_params = from_date, to_date, history_limit_per_match
            # Load train and test data for all targets
            train_matches, test_matches = load_for_training(
                COMPETITION_ID, user_token, be_params, per_page=2000, train_ratio=.75, ignore_saved=ignore_saved)

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

            is_random_search = False
            update_model = True

            hda_predictions(user_token, train_matches, test_matches, compe_data,
                            is_grid_search, is_random_search=is_random_search, update_model=update_model)

            bts_predictions(user_token, train_matches, test_matches, compe_data,
                            is_grid_search, is_random_search=is_random_search, update_model=update_model)

            over25_predictions(user_token, train_matches, test_matches, compe_data,
                               is_grid_search, is_random_search=is_random_search, update_model=update_model)

            cs_predictions(user_token, train_matches, test_matches, compe_data,
                           is_grid_search, is_random_search=is_random_search, update_model=update_model)

    print(f"\n....... END TRAIN PREDICTIONS, Happy coding! ........")


if __name__ == "__main__":
    train()
