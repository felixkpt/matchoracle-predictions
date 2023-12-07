from app.train_predictions.hda_predictions import hda_predictions
from app.train_predictions.bts_predictions import bts_predictions
from app.train_predictions.over_predictions import over_predictions
from app.train_predictions.cs_predictions import cs_predictions
from app.matches.load_matches import load_for_training
from configs.logger import Logger
import sys
from dateutil.relativedelta import relativedelta


def run_train(user_token, target, hyperparameters, PREDICTION_TYPE, COMPETITION_ID, TRAIN_TO_DATE, be_params):

    from_date = TRAIN_TO_DATE - relativedelta(years=5)
    to_date = TRAIN_TO_DATE - relativedelta(days=1)

    target = str(target).split('_')[0] if target else target

    train_to_date = TRAIN_TO_DATE.strftime("%Y-%m-%d")

    compe_data = {}
    compe_data['id'] = COMPETITION_ID
    compe_data['prediction_type'] = PREDICTION_TYPE

    Logger.info(f"Competition: {COMPETITION_ID}")
    Logger.info(f"Updating training to date: {train_to_date}\n")
    Logger.info(f"Prediction type: {PREDICTION_TYPE}\n")

    is_grid_search = False
    ignore_saved = False
    is_random_search = False
    run_score_weights = False
    target = target or 'all'
    for arg in sys.argv:
        if arg.startswith('is-grid-search'):
            is_grid_search = True
        if arg.startswith('is-random-search'):
            is_random_search = True
        if arg.startswith('ignore-saved'):
            ignore_saved = True
        if arg.startswith('target'):
            parts = arg.split('=')
            if len(parts) == 2:
                target = parts[1]
        if arg.startswith('run-score-weights'):
            run_score_weights = True

    Logger.info(f'Target: {target}\n')

    be_params = {**{'from_date': from_date, 'to_date': to_date}, **be_params}

    # Load train and test data for all targets
    train_matches, test_matches = load_for_training(
        compe_data, user_token, be_params, per_page=3000, train_ratio=.70, ignore_saved=ignore_saved)

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

    update_model = True

    if target == 'all' or target == 'hda':
        hda_predictions(user_token, train_matches, test_matches, compe_data,
                        is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)

    if target == 'all' or target == 'bts':
        bts_predictions(user_token, train_matches, test_matches, compe_data,
                        is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)

    if target == 'all' or target == 'over' or target == 'over25':
        over_predictions(user_token, train_matches, test_matches, compe_data,
                         is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)

    if target == 'all' or target == 'cs':
        cs_predictions(user_token, train_matches, test_matches, compe_data,
                       is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)
