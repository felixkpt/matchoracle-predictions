import argparse
from app.train_predictions.hda_predictions import hda_predictions
from app.train_predictions.bts_predictions import bts_predictions
from app.train_predictions.over15_predictions import over15_predictions
from app.train_predictions.over25_predictions import over25_predictions
from app.train_predictions.over35_predictions import over35_predictions
from app.train_predictions.cs_predictions import cs_predictions
from app.matches.load_matches import load_for_training
from configs.logger import Logger
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

    parser = argparse.ArgumentParser(description='Run training with different configurations.')
    parser.add_argument('--is-grid-search', action='store_true', help='Perform grid search')
    parser.add_argument('--is-random-search', action='store_true', help='Perform random search')
    parser.add_argument('--ignore-saved', action='store_true', help='Ignore saved models')
    parser.add_argument('--target', type=str, default='all', help='Specify the target')
    parser.add_argument('--run-score-weights', action='store_true', help='Run with score weights')

    args, extra_args = parser.parse_known_args()

    is_grid_search = args.is_grid_search
    is_random_search = args.is_random_search
    ignore_saved = args.ignore_saved
    run_score_weights = args.run_score_weights

    target = args.target or 'all'

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

    if target == 'all' or target == 'over15':
        over15_predictions(user_token, train_matches, test_matches, compe_data,
                         is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)

    if target == 'all' or target == 'over25':
        over25_predictions(user_token, train_matches, test_matches, compe_data,
                         is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)

    if target == 'all' or target == 'over35':
        over35_predictions(user_token, train_matches, test_matches, compe_data,
                         is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)

    if target == 'all' or target == 'cs':
        cs_predictions(user_token, train_matches, test_matches, compe_data,
                       is_grid_search, is_random_search=is_random_search, update_model=update_model, hyperparameters=hyperparameters, run_score_weights=run_score_weights)
