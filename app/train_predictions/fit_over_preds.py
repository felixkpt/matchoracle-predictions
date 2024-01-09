from configs.logger import Logger
from app.predictions_normalizers.over25_normalizer import normalizer
from app.helpers.functions import save_model, feature_importance
from app.helpers.print_results import print_preds_update_hyperparams


def fit_over_preds(user_token, model, compe_data,  target, train_frame, test_frame, FEATURES, update_model, is_grid_search, occurrences, best_params, train_matches, test_matches, hyperparameters):
    Logger.info(f"Over Target: {target}")

    model.fit(train_frame[FEATURES], train_frame[target])

    preds = model.predict(test_frame[FEATURES])
    predict_proba = model.predict_proba(test_frame[FEATURES])

    FEATURES = feature_importance(
        model, compe_data, target, FEATURES, False, 0.003) or FEATURES

    # Save model if update_model is set
    if update_model:
        save_model(model, train_frame, test_frame,
                   FEATURES, target, compe_data)

    predict_proba = normalizer(predict_proba)
        
    is_training = is_grid_search or len(hyperparameters) > 0    
    print(f'Is Training: {is_training}')
    compe_data['is_training'] = is_training
    compe_data['occurrences'] = occurrences
    compe_data['best_params'] = best_params
    compe_data['from_date'] = train_matches[0]['utc_date']
    compe_data['to_date'] = test_matches[-1]['utc_date']

    print_preds_update_hyperparams(user_token, target, compe_data,
                                   preds, predict_proba, train_frame, test_frame)
