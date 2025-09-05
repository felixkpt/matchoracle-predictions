import pandas as pd
from app.matches.load_matches import load_for_training
from app.train_predictions.train_predictions import train_predictions
from app.helpers.print_results import print_preds_hyperparams
from app.configs.active_competitions.competitions_data import update_trained_competitions
from app.configs.settings import TRAIN_MAX_CORES
from app.helpers.functions import natural_occurrences, preds_score_percentage, save_model, get_predicted_hda, get_predicted, get_predicted_cs
from app.train_predictions.hyperparameters.hyperparameters import save_hyperparameters
from app.configs.logger import Logger

def run_train(user_token, compe_data, target, be_params, prefer_saved_matches, is_grid_search, is_random_search, per_page, start_time, model_type):
    print(f'TRAIN_MAX_CORES: {TRAIN_MAX_CORES}\n')

    # Initialize configurations for model training
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
                               is_grid_search, is_random_search, update_model, total_matches, train_ratio, model_type)

    # Update competition data if there are matches processed
    if total_matches:
        compe_data['trained_to'] = be_params['to_date'].strftime('%Y-%m-%d %H:%M:%S')
        compe_data['games_counts'] = total_matches
        update_trained_competitions(user_token, compe_data, len(train_matches), start_time)
        print('\n')

def train_target_model(user_token, all_matches, compe_data, trgt, outcomes, is_grid_search, is_random_search, update_model, total_matches, train_ratio, model_type):
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
        train_matches = all_matches[:train_size]  # First x% of matches for training
        test_matches = all_matches[train_size:]   # Remaining x% for testing
 
    if total_matches < 15:
        no_enough_message(trgt, total_matches)
    else:
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
        
        # Create train and test DataFrames
        train_frame = pd.DataFrame(train_matches)
        test_frame = pd.DataFrame(test_matches)

        occurrences = natural_occurrences(
            outcomes, train_frame, test_frame, trgt)

        # Run all models and pick the best
        results = run_all_models(train_frame, test_frame, compe_data.copy(), trgt, outcomes, occurrences,
                                is_grid_search=is_grid_search, is_random_search=is_random_search, model_type=model_type)

        # Determine the best model
        model_type = max(
            results.keys(),
            key=lambda m: custom_score(results[m])
        )
        
        best_model_data = results[model_type]
        
        occurrences = best_model_data["occurrences"]
        train_frame = best_model_data["train_frame"]
        test_frame = best_model_data["test_frame"]
        preds = best_model_data["preds"]
        best_params = best_model_data["best_params"]

        if model_type == None:
            print("Could not determine best model type")
        else:
            print(f"\n🏆 Best model for {trgt}: {model_type} with F1 = {best_model_data['f1']:.3f}\n")
            natural_occurrences(outcomes, train_frame, test_frame, trgt)
            print_preds_hyperparams(trgt, model_type, compe_data, preds, best_model_data["predict_proba"], test_frame, print_minimal=True)
                
            if trgt == 'bts_target' or 'over' in trgt:
                predicted = get_predicted(preds)
            elif trgt == 'ht_hda_target' or trgt == 'ft_hda_target':
                predicted = get_predicted_hda(preds)
            elif trgt == 'cs_target':
                predicted, match_details = get_predicted_cs(preds, best_model_data["train_frame"], best_model_data["predict_proba"])
                # filter more than 0 only
                __occurrences = {}
                for k in occurrences:
                    if occurrences[k] > 0:
                        __occurrences[k] = occurrences[k]

                occurrences = __occurrences

            print("predicted::", predicted)

            compe_data['predicted'] = predicted
            compe_data['train_counts'] = len(train_frame)
            compe_data['test_counts'] = len(test_frame)
            compe_data['best_params'] = best_params
            compe_data['occurrences'] = occurrences

            compe_data['from_date'] = train_matches[0]['utc_date']
            compe_data['to_date'] = test_matches[-1]['utc_date']

            scores = preds_score_percentage(trgt, test_frame, preds, False)
            compe_data['scores'] = scores
            save_hyperparameters(model_type, compe_data, trgt, user_token)

            # Save model if update_model is set
            if update_model and best_model_data:
                save_model(model_type, best_model_data['model'], trgt, compe_data)

    print(f'***** End all models training for target: {trgt} *****\n')

def no_enough_message(trgt, total_matches):
    print(f'Not enough matches to train for target: {trgt}. Total matches available: {total_matches}')
    
# Helper function to run all models (you can reuse from previous snippet)
def run_all_models(train_frame, test_frame, compe_data, target, outcomes, occurrences, 
                   is_grid_search=False, is_random_search=False, model_type=None):

    supported_models = [
        "RandomForest",
        "BalancedRandomForestClassifier",
        "ExtraTrees",
        "GradientBoosting",
        "HistGB",
        "LogReg",
    ]

    # Validate model_type if provided
    if model_type:
        if model_type not in supported_models:
            raise ValueError(f"Unsupported model_type '{model_type}'. "
                             f"Choose from: {', '.join(supported_models)}")
        models_to_try = [model_type]
    else:
        models_to_try = supported_models

    results = {}
    for model_type in models_to_try:
        print(f"***🔹 Start Training model: {model_type}, target: {target} ***")
        try:
            model_data = train_predictions(train_frame, test_frame, compe_data.copy(),
                target, outcomes, occurrences, is_grid_search, is_random_search, model_type)
            results[model_type] = model_data
        except Exception as e:
            print(f"🛑 {model_type} failed: {e}")
            results[model_type] = None
            
    return results


def custom_score(model_data):
    if not model_data:
        return -1

    occurrences = model_data["occurrences"]
    f1 = 0.5 * model_data["f1"]
    preds = model_data["preds"]

    reward = 0
    if occurrences is not None and len(preds) > 0:
        # Calculate natural distribution
        total_occurrences = sum(occurrences.values())
        natural_dist = {cls: count / total_occurrences for cls, count in occurrences.items()}
        
        # Calculate predicted distribution
        pred_dist = {cls: list(preds).count(cls) / len(preds) for cls in set(preds)}
        
        # Calculate total variation distance (proper penalty)
        tv_distance = sum(abs(pred_dist.get(cls, 0) - natural_dist.get(cls, 0)) for cls in natural_dist) / 2
        
        # Reward should DECREASE with larger distance from natural distribution
        # dist_similarity ranges from 0 (perfect match) to 1 (complete mismatch)
        dist_similarity = 1 - tv_distance
        
        # Give moderate reward for distribution similarity
        reward = dist_similarity * 0.5
        
    print("Model:", model_data["model_type"], "Half F1:", f1, "Reward:", reward)
    return 0.7 * f1 + 0.3 * reward   # weighted
