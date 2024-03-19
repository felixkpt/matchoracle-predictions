from app.matches.load_matches import load_for_training
from app.train_predictions.hda_predictions import hda_predictions
from app.train_predictions.bts_predictions import bts_predictions
from app.train_predictions.over15_predictions import over15_predictions
from app.train_predictions.over25_predictions import over25_predictions
from app.train_predictions.over35_predictions import over35_predictions
from app.train_predictions.cs_predictions import cs_predictions
from configs.logger import Logger


def run_train(user_token, compe_data, target, be_params, ignore_saved, is_grid_search):

    # Load train and test data for all targets
    train_matches, test_matches = load_for_training(
        compe_data['id'], user_token, be_params, per_page=150, train_ratio=.75, ignore_saved=ignore_saved)

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
    
    if target is None or target == 'hda':
        hda_predictions(user_token, train_matches, test_matches, compe_data,
                        is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'bts':
        bts_predictions(user_token, train_matches, test_matches, compe_data,
                        is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over15':
        over15_predictions(user_token, train_matches, test_matches, compe_data,
                           is_grid_search, is_random_search=is_random_search, update_model=update_model)
        
    if target is None or target == 'over25':
        over25_predictions(user_token, train_matches, test_matches, compe_data,
                           is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over35':
        over35_predictions(user_token, train_matches, test_matches, compe_data,
                           is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'cs':
        cs_predictions(user_token, train_matches, test_matches, compe_data,
                       is_grid_search, is_random_search=is_random_search, update_model=update_model)
