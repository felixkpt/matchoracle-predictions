from configs.logger import Logger
from app.matches.load_matches import load_for_training
from app.train_predictions.train_predictions import train_predictions
from configs.active_competitions.competitions_data import update_trained_competitions


def run_train(user_token, compe_data, target, be_params, ignore_saved, is_grid_search, per_page):

    is_random_search = False
    update_model = True
    train_ratio = .75

    if target is None or target == 'hda' or target == 'ft-hda':
        trgt = 'ft_hda_target'
        outcomes = [0, 1, 2]
        train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, trgt, be_params, per_page, train_ratio, ignore_saved)
        # set to false since be has been accessed by this point
        ignore_saved = False

        if total_matches == 0:
            print('Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'ht-hda':
        trgt = 'ht_hda_target'
        outcomes = [0, 1, 2]
        train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, trgt, be_params, per_page, train_ratio, ignore_saved)
        # set to false since be has been accessed by this point
        ignore_saved = False

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'bts':
        trgt = 'bts_target'
        outcomes = [0, 1]
        train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, trgt, be_params, per_page, train_ratio, ignore_saved)
        # set to false since be has been accessed by this point
        ignore_saved = False

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over15':
        trgt = 'over15_target'
        outcomes = [0, 1]
        train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, trgt, be_params, per_page, train_ratio, ignore_saved)
        # set to false since be has been accessed by this point
        ignore_saved = False

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over25':
        trgt = 'over25_target'
        outcomes = [0, 1]
        train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, trgt, be_params, per_page, train_ratio, ignore_saved)
        # set to false since be has been accessed by this point
        ignore_saved = False

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over35':
        trgt = 'over35_target'
        outcomes = [0, 1]
        train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, trgt, be_params, per_page, train_ratio, ignore_saved)
        # set to false since be has been accessed by this point
        ignore_saved = False

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'cs':
        trgt = 'cs_target'
        outcomes = range(0, 121)
        train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, trgt, be_params, per_page, train_ratio, ignore_saved)
        # set to false since be has been accessed by this point
        ignore_saved = False

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    # Update trained competitions
    compe_data['trained_to'] = be_params['to_date'].strftime(
        '%Y-%m-%d %H:%M:%S')
    update_trained_competitions(user_token, compe_data, len(train_matches))


def get_matches(
        user_token, compe_data, target, be_params, per_page, train_ratio, ignore_saved):
    print(f'***** Start preds target: {target} *****')

    # Load train and test data for all targets
    train_matches, test_matches = load_for_training(
        user_token, compe_data, target, be_params, per_page, train_ratio, ignore_saved)

    total_matches = len(train_matches) + len(test_matches)

    return train_matches, test_matches, total_matches
