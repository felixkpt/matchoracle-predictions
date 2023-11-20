from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from configs.settings import PREDICTORS
from app.train_predictions.tuning.cs_target.cs_tuning import HYPER_PARAMS
from app.predictions_normalizers.cs_normalizer import normalizer
from app.train_predictions.includes.functions import save_model
from app.train_predictions.tuning.cs_target.cs_grid_search import grid_search
from app.helpers.print_results import print_cs_predictions as print_predictions


def cs_predictions_normalizer(train_matches, test_matches, COMPETITION_ID, do_grid_search=False, update_model=False):

    target = 'cs_target'
    scoring = 'weighted'
    grid_search_scoring = 'precision_weighted'

    Logger.info(f"Prediction Target: {target}")

    # Select the appropriate class weight dictionary based on the target
    has_weights = True
    hyper_params = HYPER_PARAMS[0]
    class_weight = hyper_params['class_weight']
    min_samples_split_ratio = hyper_params['min_samples_split_ratio']

    len_train = len(train_matches)
    n_estimators = int(0.2 * len_train)
    min_samples_split = int(min_samples_split_ratio * len_train)
    min_samples_split = min_samples_split if min_samples_split > 2 else 2

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {hyper_params}\n")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    # Calculate the percentages
    total_matches = len(train_matches)

    occurances = []

    model = RandomForestClassifier(random_state=1, n_estimators=n_estimators, class_weight=class_weight,
                                   min_samples_split=min_samples_split)

    if do_grid_search or not has_weights:
        grid_search(model, train_frame, target, occurances, COMPETITION_ID, True)
        return

    # Save model if update_model is set
    if update_model:
        save_model(model, train_frame, test_frame,
                   PREDICTORS, target, COMPETITION_ID)

    model.fit(train_frame[PREDICTORS], train_frame[target])

    # Make predictions on the test data
    preds = model.predict(test_frame[PREDICTORS])
    predict_proba = model.predict_proba(test_frame[PREDICTORS])

    predict_proba = normalizer(predict_proba)

    print_predictions(target, total_matches, test_frame, preds,
                      predict_proba, scoring, occurances, print_minimal=True)

    return [preds, predict_proba]
