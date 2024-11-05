from app.matches.load_matches import load_for_training
from app.train_predictions.train_predictions import train_predictions
from app.configs.active_competitions.competitions_data import update_trained_competitions


def run_train(user_token, compe_data, target, be_params, ignore_saved_matches, is_grid_search, per_page, start_time):
    
    # Initialize configurations for model training
    is_random_search = False  # Indicates if random search (hyperparameter tuning) should be used
    update_model = True       # Flag to determine if the model should be updated after training
    train_ratio = 0.75        # Defines the training data proportion (75% train, 25% test)

    # Load all matches for training based on parameters provided
    all_matches = load_for_training(user_token, compe_data, be_params, per_page, ignore_saved_matches)

    # Calculate number of matches and split data into training and testing sets
    total_matches = len(all_matches)
    train_size = int(total_matches * train_ratio)
    train_matches = all_matches[:train_size]  # First 75% of matches for training
    test_matches = all_matches[train_size:]   # Remaining 25% for testing

    # Define and train model based on target variable specified
    if target is None or target == 'hda' or target == 'ft-hda':
        trgt = 'ft_hda_target'  # Target for full-time home/draw/away predictions
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1, 2]    # Possible outcomes: home win, draw, away win

        all_matches = [m for m in all_matches if m['ht_hda_target'] >= 0]
        total_matches = len(all_matches)

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            # Train predictions model with the specified parameters
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    # Similar blocks follow for each specific target type with respective outcomes and training
    if target is None or target == 'ht-hda':
        trgt = 'ht_hda_target'  # Target for half-time home/draw/away predictions
        print(f'***** Start preds target: {target} *****')
        # Filter matches to only include those with a valid half-time target
        all_matches = [m for m in all_matches if m['ht_hda_target'] >= 0]
        total_matches = len(all_matches)
        outcomes = [0, 1, 2]  # Possible outcomes for half-time predictions

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    # Targets for "Both Teams to Score", "Over 1.5 Goals", "Over 2.5 Goals", etc.
    # Each block defines its unique target, outcomes, and calls train_predictions function
    if target is None or target == 'bts':
        trgt = 'bts_target'  # Both Teams to Score target
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]     # Possible outcomes: No (0), Yes (1)

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over15':
        trgt = 'over15_target'  # Over 1.5 goals target
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]       # Possible outcomes: No (0), Yes (1)

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over25':
        trgt = 'over25_target'  # Over 2.5 goals target
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]       # Possible outcomes: No (0), Yes (1)

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'over35':
        trgt = 'over35_target'  # Over 3.5 goals target
        print(f'***** Start preds target: {target} *****')
        outcomes = [0, 1]       # Possible outcomes: No (0), Yes (1)

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    if target is None or target == 'cs':
        trgt = 'cs_target'      # Correct score target
        print(f'***** Start preds target: {target} *****')
        outcomes = range(0, 121) # Possible outcomes: range of correct scores

        if total_matches == 0:
            print(f'Aborting, no matches to make predictions for {trgt}.\n')
        else:
            train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                              is_grid_search, is_random_search=is_random_search, update_model=update_model)

    # Update competition data with the training information if there are matches processed
    if total_matches:
        compe_data['trained_to'] = be_params['to_date'].strftime('%Y-%m-%d %H:%M:%S')
        compe_data['games_counts'] = total_matches
        update_trained_competitions(user_token, compe_data, len(train_matches), start_time)
