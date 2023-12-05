from app.realtime_model_metrics import RealtimeModelMetrics


def realtime_metrics(user_token):

    target = 'hda_target'
    target = 'bts_target'
    target = 'over25_target'
    # target = 'cs_target'

    prediction_types = [
        # 'regular_prediction',
        'regular_prediction_last_5',
        'regular_prediction_last_7',
        'regular_prediction_last_10',
        'regular_prediction_last_15',
        # 'regular_prediction_last_20',
        # 'regular_prediction_last_25',
    ]

    model_metrics = RealtimeModelMetrics(user_token, target, prediction_types)
    comparison_results = model_metrics.compare_models()
    model_metrics.choose_best_model(comparison_results)
