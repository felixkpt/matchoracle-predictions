from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from app.helpers.functions import natural_occurrences, save_model, get_features, feature_importance
from app.train_predictions.hyperparameters.hyperparameters import get_hyperparameters
from app.helpers.print_results import print_preds_update_hyperparams
import numpy as np

# Import grid_search and normalizer
# Grid search imports
from app.train_predictions.tuning.hda_target.hda_grid_search import grid_search as hda_grid_search
from app.train_predictions.tuning.bts_target.bts_grid_search import grid_search as bts_grid_search
from app.train_predictions.tuning.over15_target.over15_grid_search import grid_search as over15_grid_search
from app.train_predictions.tuning.over25_target.over25_grid_search import grid_search as over25_grid_search
from app.train_predictions.tuning.over35_target.over35_grid_search import grid_search as over35_grid_search
from app.train_predictions.tuning.cs_target.cs_grid_search import grid_search as cs_grid_search
# Normalizer imports
from app.predictions_normalizers.hda_normalizer import normalizer as hda_normalizer
from app.predictions_normalizers.bts_normalizer import normalizer as bts_normalizer
from app.predictions_normalizers.over_normalizer import normalizer as over_normalizer
from app.predictions_normalizers.cs_normalizer import normalizer as cs_normalizer


# Set a random seed for reproducibility
np.random.seed(42)


def train_predictions(user_token, train_matches, test_matches, compe_data, target, outcomes, is_grid_search=False, is_random_search=False, update_model=False):

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

    features, has_features = get_features(compe_data, target)
    FEATURES = features
    print(f"Has filtered features: {'Yes' if has_features else 'No'}")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    occurrences = natural_occurrences(
        outcomes, train_frame, test_frame, target)

    if occurrences[0] == 0:
        print(f'No results in this category.')
        return None

    # Select the appropriate class weight dictionary based on the target
    hyper_params, has_weights = get_hyperparameters(
        compe_data, target, outcomes)

    model = RandomForestClassifier(**hyper_params)

    best_params = None
    if is_grid_search or not has_weights:
        is_grid_search = True
        if target == 'ft_hda_target' or target == 'ht_hda_target':
            grid_search_function = hda_grid_search
        elif target == 'bts_target':
            grid_search_function = bts_grid_search
        elif target == 'over15_target':
            grid_search_function = over15_grid_search
        elif target == 'over25_target':
            grid_search_function = over25_grid_search
        elif target == 'over35_target':
            grid_search_function = over35_grid_search
        elif target == 'cs_target':
            grid_search_function = cs_grid_search
        else:
            grid_search_function = None  # Default grid search function

        # should handle case when is_grid_search is True and grid_search_function is missing
        if grid_search_function:
            best_params = grid_search_function(
                model, train_frame, FEATURES, target, occurrences, is_random_search)

            hyper_params = best_params
            model.set_params(**hyper_params)

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {hyper_params}\n")

    model.fit(train_frame[FEATURES], train_frame[target])

    FEATURES = feature_importance(
        model, compe_data, target, FEATURES, False, 0.008)

    # Save model if update_model is set
    if update_model:
        save_model(model, train_frame, test_frame,
                   FEATURES, target, compe_data)

    # Make predictions on the test data
    preds = model.predict(test_frame[FEATURES])
    predict_proba = model.predict_proba(test_frame[FEATURES])

    # Choose normalizer based on target
    if target == 'ft_hda_target' or target == 'ht_hda_target':
        normalizer_function = hda_normalizer
    elif target == 'bts_target':
        normalizer_function = bts_normalizer
    elif target == 'over15_target':
        normalizer_function = over_normalizer
    elif target == 'over25_target':
        normalizer_function = over_normalizer
    elif target == 'over35_target':
        normalizer_function = over_normalizer
    elif target == 'cs_target':
        normalizer_function = cs_normalizer
    else:
        normalizer_function = None  # Default normalizer function

    if normalizer_function:
        predict_proba = normalizer_function(predict_proba)

    compe_data['is_training'] = is_grid_search
    compe_data['occurrences'] = occurrences
    compe_data['best_params'] = best_params
    compe_data['from_date'] = train_matches[0]['utc_date']
    compe_data['to_date'] = test_matches[-1]['utc_date']

    print_preds_update_hyperparams(user_token, target, compe_data, preds,
                                   predict_proba, train_frame, test_frame, print_minimal=True)

    print(f'***** End preds target: {target} *****\n')

    return [preds, predict_proba, occurrences]
