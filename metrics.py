from app.model_metrics import ModelMetrics


def metrics(user_token):

    target = 'hda_target'
    # target = 'bts_target'
    # target = 'over25_target'
    # target = 'cs_target'

    prediction_types = [
        'regular_prediction_last_5_matches_optimized',
        'regular_prediction_last_7_matches_optimized',
        'regular_prediction_last_10_matches_optimized',
        'regular_prediction_last_15_matches_optimized',
        'regular_prediction_last_20_matches_optimized',
        'regular_prediction_last_25_matches_optimized',
        'regular_prediction_last_5_matches_optimized_30',
        'regular_prediction_last_7_matches_optimized_30',
        'regular_prediction_last_10_matches_optimized_30',
        'regular_prediction_last_15_matches_optimized_30',
        'regular_prediction_last_20_matches_optimized_30',
        'regular_prediction_last_30_matches_optimized_30',
    ]

    model_metrics = ModelMetrics(target, prediction_types)
    comparison_results = model_metrics.compare_models()
    model_metrics.choose_best_model(comparison_results)
