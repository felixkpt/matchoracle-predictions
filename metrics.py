import argparse
from app.model_metrics import ModelMetrics
from configs.logger import Logger

def metrics(user_token):

    parser = argparse.ArgumentParser(description='Run model metrics for a specified target and competition.')
    parser.add_argument('--target', choices=['hda', 'bts', 'over15', 'over25', 'over35', 'cs'], help='Target for model metrics')
    parser.add_argument('--competition', type=int, help='Competition ID for model metrics')

    args, extra_args = parser.parse_known_args()

    target = args.target or 'hda'
    competition_id = args.competition

    Logger.info(f'Target: {target}\n')

    prediction_types = [
        'regular_prediction_7_4_4',
        'regular_prediction_10_4_4',
        'regular_prediction_10_6_4',
        'regular_prediction_10_6_6',
    ]

    target = f'{target}_target'

    competitions = [competition_id] if competition_id is not None else [25, 47, 48, 125, 148]
    model_metrics = ModelMetrics(target, prediction_types, competitions)
    comparison_results = model_metrics.compare_models()
    model_metrics.choose_best_model(comparison_results)
