import pandas as pd
from configs.logger import Logger
from configs.settings import COMMON_PREDICTORS
from app.predictions_normalizers.cs_normalizer import normalizer
from app.train_predictions.includes.functions import get_model
from app.helpers.print_results import print_preds_update_hyperparams


def cs_predictions(user_token, matches, COMPETITION_ID):

    target = 'cs_target'
    PREDICTORS = COMMON_PREDICTORS + ['over25_target', 'hda_target']

    Logger.info(f"Prediction Target: {target}")

    # Create train and test DataFrames
    predict_frame = pd.DataFrame(matches)

    # Get the model
    model = get_model(target, COMPETITION_ID)

    # Make predictions on the test data
    preds = model.predict(predict_frame[PREDICTORS])
    predict_proba = model.predict_proba(predict_frame[PREDICTORS])

    predict_proba = normalizer(predict_proba)

    print_preds_update_hyperparams(target, user_token, COMPETITION_ID, target, preds, predict_proba,
                                   train_frame=None, test_frame=predict_frame, occurrences=None, best_params=None)

    return [preds, predict_proba]
