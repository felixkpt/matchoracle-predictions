from app.model_metrics import ModelMetrics


def metrics(user_token):

    target = 'hda_target'
    # target = 'bts_target'
    # target = 'over15_target'
    target = 'over25_target'
    # target = 'over35_target'
    # target = 'cs_target'

    prediction_types = [
        'regular_prediction_10_4_4',
        'regular_prediction_10_4_6',
        'regular_prediction_10_4_8',
        'regular_prediction_10_6_4',
        'regular_prediction_10_6_6',
        'regular_prediction_10_6_8',
        'regular_prediction_15_4_4',
    ]

    model_metrics = ModelMetrics(target, prediction_types)
    comparison_results = model_metrics.compare_models()
    model_metrics.choose_best_model(comparison_results)
