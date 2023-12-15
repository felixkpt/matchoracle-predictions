from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from app.train_predictions.tuning.over15_target.over15_grid_search import grid_search
from app.train_predictions.tuning.prepare_grid_search import prepare_grid_search
from app.helpers.functions import natural_occurrences, get_features
from app.train_predictions.hyperparameters.hyperparameters import get_hyperparameters
from app.train_predictions.fit_over_preds import fit_over_preds


def over15_predictions(user_token, train_matches, test_matches, compe_data, is_grid_search=False, is_random_search=False, update_model=False, hyperparameters={}, run_score_weights=False):

    target = 'over15_target'

    Logger.info(f"Prediction Target: {target}")

    features, has_features = get_features(compe_data, target, is_grid_search)
    FEATURES = features
    print(f"Has filtered features: {'Yes' if has_features else 'No'}")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    outcomes = [0, 1]
    occurrences = natural_occurrences(
        outcomes, train_frame, test_frame, target)

    # Select the appropriate class weight dictionary based on the target
    hyper_params, has_weights = get_hyperparameters(
        compe_data, target)

    hyper_params = {**hyper_params, **hyperparameters}
    model = RandomForestClassifier(**hyper_params)

    best_params = None
    if is_grid_search or not has_weights:
        is_grid_search = True
        best_params = prepare_grid_search(grid_search, compe_data,
                                          model, train_frame, FEATURES, target, occurrences, is_random_search, run_score_weights)

        hyper_params = best_params
        n_estimators_fraction = hyper_params['n_estimators_fraction']
        hyper_params.pop('n_estimators_fraction')

        model.set_params(**hyper_params)

        best_params['n_estimators_fraction'] = n_estimators_fraction

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {hyper_params}\n")

    if hyperparameters:
        best_params = hyper_params

    occurrences = natural_occurrences(
        outcomes, train_frame, test_frame, target)

    fit_over_preds(user_token, model, compe_data,  target, train_frame, test_frame, FEATURES,
                    update_model, is_grid_search, occurrences, best_params, train_matches, test_matches, hyperparameters)
