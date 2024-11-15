from app.matches.load_matches import load_for_training
from app.train_predictions.train_predictions import train_predictions
from app.configs.active_competitions.competitions_data import update_trained_competitions

def run_train(user_token, compe_data, target, be_params, prefer_saved_matches, is_grid_search, per_page, start_time):
    # Initialize configurations for model training
    is_random_search = False  
    update_model = True       
    train_ratio = 0.75     

    target = 'ft-hda' if target =='hda' else target

    # Define target configurations in a dictionary
    target_configs = {
        'hda': {'trgt': 'ft_hda_target', 'outcomes': [0, 1, 2]},
        'ht-hda': {'trgt': 'ht_hda_target', 'outcomes': [0, 1, 2]},
        'bts': {'trgt': 'bts_target', 'outcomes': [0, 1]},
        'over15': {'trgt': 'over15_target', 'outcomes': [0, 1]},
        'over25': {'trgt': 'over25_target', 'outcomes': [0, 1]},
        'over35': {'trgt': 'over35_target', 'outcomes': [0, 1]},
        'cs': {'trgt': 'cs_target', 'outcomes': range(0, 121)},
    }

    # Check for invalid target and handle it
    if target and target not in target_configs:
        raise ValueError(f"Invalid target: {target}")

    # Load all matches for training
    all_matches = load_for_training(user_token, compe_data, be_params, per_page, prefer_saved_matches)
    total_matches = len(all_matches)
    train_size = int(total_matches * train_ratio)
    train_matches = all_matches[:train_size]

    print(f'Total matches before filters: {total_matches}\n')


    # Train models for each target if the target matches or is unspecified
    for key, config in target_configs.items():
        if target is None or target == key:
            train_target_model(user_token, all_matches, compe_data, config['trgt'], config['outcomes'],
                               is_grid_search, is_random_search, update_model, total_matches, train_ratio)

    # Update competition data if there are matches processed
    if total_matches:
        compe_data['trained_to'] = be_params['to_date'].strftime('%Y-%m-%d %H:%M:%S')
        compe_data['games_counts'] = total_matches
        update_trained_competitions(user_token, compe_data, len(train_matches), start_time)
        print('\n')

def train_target_model(user_token, all_matches, compe_data, trgt, outcomes, is_grid_search, is_random_search, update_model, total_matches, train_ratio):
    print(f'***** Start preds target: {trgt} *****')

    if (trgt != 'ht_hda_target'):
        train_size = int(total_matches * train_ratio)
        train_matches, test_matches = all_matches[:train_size], all_matches[train_size:]
    else:
       # Filter matches to only include those with a valid half-time target
        all_matches = [m for m in all_matches if m['ht_hda_target'] >= 0]

        # Calculate number of matches and split data into training and testing sets
        total_matches = len(all_matches)
        train_size = int(total_matches * train_ratio)
        train_matches = all_matches[:train_size]  # First 75% of matches for training
        test_matches = all_matches[train_size:]   # Remaining 25% for testing
 
    if total_matches < 15:
        no_enough_message(trgt, total_matches)
    else:
        train_predictions(user_token, train_matches, test_matches, compe_data, trgt, outcomes,
                          is_grid_search, is_random_search=is_random_search, update_model=update_model)

def no_enough_message(trgt, total_matches):
    print(f'Not enough matches to train for target: {trgt}. Total matches available: {total_matches}')