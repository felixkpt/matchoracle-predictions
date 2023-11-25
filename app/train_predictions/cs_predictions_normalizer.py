from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from configs.settings import COMMON_PREDICTORS
from app.predictions_normalizers.cs_normalizer import normalizer
from app.train_predictions.tuning.cs_target.cs_grid_search import grid_search
from app.train_predictions.includes.functions import natural_occurrences, save_model
from app.train_predictions.hyperparameters.hyperparameters import get_hyperparameters
from app.helpers.print_results import print_preds_update_hyperparams
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)


def cs_predictions_normalizer(user_token, train_matches, test_matches, compe_data, do_grid_search=False, is_random_search=False, update_model=False):
    return
    target = 'cs_unsensored_target'
    PREDICTORS = COMMON_PREDICTORS + \
        ['over25_target', 'hda_target', 'bts_target']

    Logger.info(f"Prediction Target: {target}")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    outcomes = range(0, 121)
    occurrences = natural_occurrences(
        outcomes, train_frame, test_frame, target)

   # Select the appropriate class weight dictionary based on the target
    hyper_params, has_weights = get_hyperparameters(
        compe_data, target, outcomes)

    model = RandomForestClassifier(**hyper_params)

    best_params = None
    if do_grid_search or not has_weights:
        best_params = grid_search(
            model, train_frame, PREDICTORS, target, occurrences, is_random_search)

        hyper_params = best_params
        model.set_params(**hyper_params)

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {hyper_params}\n")

    # Save model if update_model is set
    if update_model:
        save_model(model, train_frame, test_frame,
                   PREDICTORS, target, compe_data['id'])

    model.fit(train_frame[PREDICTORS], train_frame[target])

    # Make predictions on the test data
    preds = model.predict(test_frame[PREDICTORS])
    predict_proba = model.predict_proba(test_frame[PREDICTORS])

    predict_proba = normalizer(predict_proba)

    compe_data['occurrences'] = occurrences
    compe_data['best_params'] = best_params
    compe_data['from_date'] = train_matches[0]['utc_date']
    compe_data['to_date'] = test_matches[-1]['utc_date']

    print_preds_update_hyperparams(user_token, target, compe_data,
                                   preds, predict_proba, train_frame, test_frame, print_minimal=False)

    return [preds, predict_proba, occurrences]
