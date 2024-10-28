from app.matches.load_matches import load_for_training
from app.train_predictions.train_predictions import train_predictions
from app.configs.active_competitions.competitions_data import update_trained_competitions


def run_train(user_token, compe_data, target, be_params, ignore_saved_matches, is_grid_search, per_page, start_time):

    is_random_search = False
    update_model = True
    train_ratio = .75

    train_matches, test_matches, total_matches = get_matches(
            user_token, compe_data, be_params, per_page, train_ratio, ignore_saved_matches)

    if target is None or target == 'hda' or target == 'ft-hda':
        trgt = 'ft_hda_target'
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1, 2]

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'ht-hda':
        trgt = 'ht_hda_target'
        print(f'***** Start preds target: {target} *****')
        all_matches = [m for m in all_matches if m['ht_hda_target'] >= 0]
        outcomes = [0, 1, 2]


        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'bts':
        trgt = 'bts_target'
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over15':
        trgt = 'over15_target'
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over25':
        trgt = 'over25_target'
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over35':
        trgt = 'over35_target'
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'cs':
        trgt = 'cs_target'
        print(f'***** Start preds target: {target} *****')
        outcomes = range(0, 121)

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if total_matches:
        # Update trained competitions
        compe_data['trained_to'] = be_params['to_date'].strftime(
            '%Y-%m-%d %H:%M:%S')
        compe_data['games_counts'] = total_matches
        update_trained_competitions(user_token, compe_data, len(train_matches), start_time)


def get_matches(user_token, compe_data, be_params, per_page, train_ratio, ignore_saved_matches):

    # Load train and test data for all targets
    all_matches = load_for_training(user_token, compe_data, be_params, per_page, ignore_saved_matches)

    total_matches = len(all_matches)
    train_size = int(total_matches * train_ratio)

    # Split matches into train and test sets
    train_matches = all_matches[:train_size]
    test_matches = all_matches[train_size:]


    return train_matches, test_matches, total_matches
