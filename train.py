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
# championship, epl, portugal primera, laliga
# 47, 48, 125, 148
COMPETITION_ID = 47

# Calculate from_date and to_date
TRAIN_TO_DATE = datetime.strptime('2023-11-25', '%Y-%m-%d')
from_date = TRAIN_TO_DATE - relativedelta(months=48)
to_date = TRAIN_TO_DATE - relativedelta(days=1)


def train(user_token):
    print("\n............... START TRAIN PREDICTIONS ..................\n")

    train_to_date = TRAIN_TO_DATE.strftime("%Y-%m-%d")

    Logger.info(f"Competition: {COMPETITION_ID}")
    Logger.info(f"Date: {train_to_date}\n")

    is_grid_search = False
    for arg in sys.argv:
        if arg.startswith('grid-search'):
            is_grid_search = True

    # Load train and test data for all targets
    train_matches, test_matches = load_for_training(
        COMPETITION_ID=COMPETITION_ID, from_date=from_date, to_date=to_date, user_token=user_token, per_page=2000, train_ratio=.70, ignore_saved=False)

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

    # hda_predictions(train_matches, test_matches, COMPETITION_ID, is_grid_search, True)

    bts_predictions(
        train_matches, test_matches, COMPETITION_ID, is_grid_search, True)

    # over25_predictions(
    #     train_matches, test_matches, COMPETITION_ID, is_grid_search, True)

    # cs_predictions(
    #     train_matches, test_matches, COMPETITION_ID, is_grid_search, True)

    # cs_predictions_normalizer(COMPETITION_ID, is_grid_search, True)

    print(f"\n....... END TRAIN PREDICTIONS, Happy coding! ........")


if __name__ == "__main__":
    train()
