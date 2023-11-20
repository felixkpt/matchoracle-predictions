import pandas as pd
from configs.logger import Logger
from configs.settings import PREDICTORS
from app.predictions_normalizers.bts_normalizer import normalizer
from app.train_predictions.includes.functions import get_model
from app.helpers.print_results import print_bts_predictions as print_predictions


def bts_predictions(matches, COMPETITION_ID):

    target = 'bts_target'
    scoring = 'weighted'

    Logger.info(f"Prediction Target: {target}")

    # Create train and test DataFrames
    matches_frame = pd.DataFrame(matches)
    # print(matches_frame)
    total_matches = len(matches_frame)

    # Get the model
    model = get_model(target, COMPETITION_ID)

    # Make predictions on the test data
    preds = model.predict(matches_frame[PREDICTORS])
    predict_proba = model.predict_proba(matches_frame[PREDICTORS])

    predict_proba = normalizer(predict_proba)

    print_predictions(target, total_matches, matches_frame,
                            preds, predict_proba, scoring)

    return [preds, predict_proba]
