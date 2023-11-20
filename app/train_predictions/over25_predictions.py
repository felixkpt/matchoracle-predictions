from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from configs.settings import PREDICTORS
from app.predictions_normalizers.over25_normalizer import normalizer
from app.train_predictions.tuning.over25_target.over25_grid_search import grid_search
from app.train_predictions.includes.functions import natural_occurances, save_model, get_hyperparameters
from app.helpers.print_results import print_over25_predictions as print_predictions


def over25_predictions(train_matches, test_matches, COMPETITION_ID, do_grid_search=False, update_model=False):

    target = 'over25_target'
    scoring = 'weighted'

    Logger.info(f"Prediction Target: {target}")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    outcomes = [0, 1]
    occurances = natural_occurances(outcomes, train_frame, target)

   # Select the appropriate class weight dictionary based on the target
    n_estimators, min_samples_split, class_weight, has_weights = get_hyperparameters(
        COMPETITION_ID, target, outcomes=outcomes)

    model = RandomForestClassifier(random_state=1, n_estimators=n_estimators, class_weight=class_weight,
                                   min_samples_split=min_samples_split)
    
    if do_grid_search or not has_weights:
        grid_search(model, train_frame, target,
                    occurances, COMPETITION_ID, True)
        n_estimators, min_samples_split, class_weight, has_weights = get_hyperparameters(
            COMPETITION_ID, target)
        occurances = natural_occurances(outcomes, train_frame, target)

        model = RandomForestClassifier(random_state=1, n_estimators=n_estimators, class_weight=class_weight,
                                       min_samples_split=min_samples_split)

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {n_estimators, class_weight, min_samples_split}\n")

    # Save model if update_model is set
    if update_model:
        save_model(model, train_frame, test_frame,
                   PREDICTORS, target, COMPETITION_ID)

    model.fit(train_frame[PREDICTORS], train_frame[target])

    # Make predictions on the test data
    preds = model.predict(test_frame[PREDICTORS])
    predict_proba = model.predict_proba(test_frame[PREDICTORS])

    predict_proba = normalizer(predict_proba)

    print_predictions(target, test_frame, preds,
                      predict_proba, scoring, occurances, True)

    return [preds, predict_proba, occurances]
