import pandas as pd
from app.configs.logger import Logger
from app.predictions_normalizers.bts_normalizer import normalizer
from app.helpers.functions import get_model, get_features
from app.helpers.print_results import print_preds_hyperparams


def bts_predictions(matches, compe_data):

    target = 'bts_target'
    
    Logger.info(f"Prediction Target: {target}")

    # Try getting the model
    try:
        model, model_type = get_model(target, compe_data)
    except FileNotFoundError:
        return [None, None]

    FEATURES, has_features = get_features(model_type, compe_data, target)

    # If there are no valid features, return None
    if not has_features:
        Logger.error("No filtered features found for predictions.")
        return [None, None]

    # Create train and test DataFrames
    predict_frame = pd.DataFrame(matches)

   # Make predictions on the test data
    preds = model.predict(predict_frame[FEATURES])
    predict_proba = model.predict_proba(predict_frame[FEATURES])

    predict_proba = normalizer(predict_proba)

    print_preds_hyperparams(target, model_type, compe_data, preds, predict_proba, test_frame=predict_frame, print_minimal=False)

    return [preds, predict_proba]
