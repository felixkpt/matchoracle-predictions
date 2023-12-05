from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from app.predictions_normalizers.cs_normalizer import normalizer
from app.train_predictions.tuning.cs_normalizer_target.cs_normalizer_target import grid_search
from app.helpers.functions import natural_occurrences, save_model, get_features, feature_importance
from app.train_predictions.hyperparameters.hyperparameters import get_hyperparameters
from app.helpers.print_results import print_preds_update_hyperparams
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)


def cs_predictions_normalizer(user_token, train_matches, test_matches, compe_data, is_grid_search=False, is_random_search=False, update_model=False):
    
    target = 'cs_target'

    Logger.info(f"Prediction Target: {target}")
    
    features, has_features = get_features(compe_data, target, is_grid_search, True)
    FEATURES = features
    print(f"Has filtered features: {'Yes' if has_features else 'No'}")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    outcomes = range(0, 121)
    occurrences = natural_occurrences(
        outcomes, train_frame, test_frame, target)

   # Select the appropriate class weight dictionary based on the target
    hyper_params, has_weights = get_hyperparameters(
        compe_data, target, True)

    hyper_params = {**hyper_params, **hyperparameters}
    model = RandomForestClassifier(**hyper_params)

    best_params = None
    if is_grid_search or not has_weights:
        is_grid_search = True
        best_params = grid_search(
            model, train_frame, FEATURES, target, occurrences, is_random_search)

        hyper_params = best_params
        n_estimators_fraction = hyper_params['n_estimators_fraction']
        hyper_params.pop('n_estimators_fraction')

        model.set_params(**hyper_params)

        best_params['n_estimators_fraction'] = n_estimators_fraction

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {hyper_params}\n")

    model.fit(train_frame[FEATURES], train_frame[target])

    # Make predictions on the test data
    preds = model.predict(test_frame[FEATURES])
    predict_proba = model.predict_proba(test_frame[FEATURES])

    FEATURES = feature_importance(
        model, compe_data, target, FEATURES, False, 0.004, True)

    # Save model if update_model is set
    if update_model:
        save_model(model, train_frame, test_frame,
                   FEATURES, target, compe_data)

    predict_proba = normalizer(predict_proba)

    if hyperparameters:
        best_params = hyper_params
        
    is_training = is_grid_search or len(hyperparameters) > 0    
    print(f'Is Training: {is_training}')
    compe_data['is_training'] = is_training
    compe_data['occurrences'] = occurrences
    compe_data['best_params'] = best_params
    compe_data['from_date'] = train_matches[0]['utc_date']
    compe_data['to_date'] = test_matches[-1]['utc_date']
    compe_data['is_normalizer'] = True

    print_preds_update_hyperparams(user_token, target, compe_data,
                                   preds, predict_proba, train_frame, test_frame, print_minimal=False)

    return [preds, predict_proba, occurrences]
