from app.matches.load_matches import load_for_training
from app.train_predictions.train_predictions import train_predictions
from app.helpers.print_results import print_preds_update_hyperparams
from app.configs.active_competitions.competitions_data import update_trained_competitions
from app.configs.settings import TRAIN_MAX_CORES
from app.helpers.functions import natural_occurrences

def run_train(user_token, compe_data, target, be_params, prefer_saved_matches, is_grid_search, per_page, start_time):
    print(f'TRAIN_MAX_CORES: {TRAIN_MAX_CORES}\n')

    # Initialize configurations for model training
    is_random_search = False  
    update_model = True       
    train_ratio = 0.70   

    # Define target configurations in a dictionary
    target_configs = {
        'ft-hda': {'trgt': 'ft_hda_target', 'outcomes': [0, 1, 2]},
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
    print(f'***** Start all models training for target: {trgt} *****')

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
        # Run all models and pick the best
        results = run_all_models(user_token, train_matches, test_matches, compe_data.copy(), trgt, outcomes,
                                is_grid_search=is_grid_search, is_random_search=is_random_search, update_model=update_model)

        # Determine the best model based on F1 score
        best_model_type = max(
            results.keys(),
            key=lambda m: results[m]['f1'] if results[m] else -1
        )
        best_model_data = results[best_model_type]

        train_frame = best_model_data["train_frame"]
        test_frame = best_model_data["test_frame"]
        

        print(f"\n🏆 Best model for {trgt}: {best_model_type} with F1 = {best_model_data['f1']:.3f}\n")

        natural_occurrences(outcomes, train_frame, test_frame, trgt)
        print_preds_update_hyperparams(user_token, trgt, best_model_type, compe_data, best_model_data["preds"],
                                        best_model_data["predict_proba"], train_frame, test_frame, print_minimal=True)

        # Optionally save the best model
        if update_model and best_model_data:
            save_model(best_model_data['model'], train_matches, test_matches,
                    best_model_data['features'], trgt, compe_data)

    print(f'***** End all models training for target: {trgt} *****\n')

def no_enough_message(trgt, total_matches):
    print(f'Not enough matches to train for target: {trgt}. Total matches available: {total_matches}')
    
# Helper function to run all models (you can reuse from previous snippet)
def run_all_models(user_token, train_matches, test_matches, compe_data, target, outcomes,
                   is_grid_search=False, is_random_search=False, update_model=False):

    models_to_try = ["RandomForest", "ExtraTrees", "GradientBoosting", "HistGB", "LogReg"]

    results = {}
    for model_type in models_to_try:
        print(f"\n🔹 Training model: {model_type}")
        try:
            model_data = train_predictions(
                user_token, train_matches, test_matches, compe_data.copy(),
                target, outcomes, is_grid_search, is_random_search,
                update_model, model_type
            )
            results[model_type] = model_data
        except Exception as e:
            print(f"🛑 {model_type} failed: {e}")
            results[model_type] = None
            
    return results


def save_model(model, train_matches, test_matches, features, trgt, compe_data):
    pass
