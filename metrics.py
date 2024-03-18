import os
import json


def metrics(target):
    prediction_types = [
        'regular_prediction_last_5_matches_optimized',
        'regular_prediction_last_7_matches_optimized',
        'regular_prediction_last_10_matches_optimized',
        'regular_prediction_last_15_matches_optimized',
        'regular_prediction_last_20_matches_optimized',
        'regular_prediction_last_25_matches_optimized',
    ]

    base_path = f"/home/felix/Documents/DEV/PY/matchoracle-predictions-v2/app/train_predictions/hyperparameters/"
    metrics = load_metrics(base_path, target, prediction_types)
    comparison_results = compare_models(metrics)
    choose_best_model(target, comparison_results)


def load_metrics(base_path, target, prediction_types):
    metrics = {}
    for prediction_type in prediction_types:
        json_path = os.path.join(
            base_path, f'{prediction_type}/{target}_hyperparameters.json')
        with open(json_path, 'r') as file:
            metrics[prediction_type] = json.load(file)
    return metrics


def compare_models(metrics):
    comparison_results = {}

    for prediction_type, competition_metrics in metrics.items():
        competition_metrics_data = {}

        for competition_id, metrics_data in competition_metrics.items():
            accuracy_score = metrics_data['accuracy_score']
            precision_score = metrics_data['precision_score']
            f1_score = metrics_data['f1_score']
            average_score = metrics_data['average_score']

            competition_metrics_data[competition_id] = {
                'accuracy_score': accuracy_score,
                'precision_score': precision_score,
                'f1_score': f1_score,
                'average_score': average_score
            }

        comparison_results[prediction_type] = competition_metrics_data

    return comparison_results


def choose_best_model(target, comparison_results):
    total_average_scores = {}

    for prediction_type, competition_metrics in comparison_results.items():
        total_average_score = sum(
            metrics_data['average_score'] for metrics_data in competition_metrics.values()
        )

        num_competitions = len(competition_metrics)
        average_percentage = total_average_score / num_competitions

        total_average_scores[prediction_type] = average_percentage

    best_prediction_type = max(
        total_average_scores, key=total_average_scores.get)

    print(
        f"\nBest model for Target {target}:\n")
    print(f"Chosen Prediction Type: {best_prediction_type}")
    print(
        f"Average Percentage: {total_average_scores[best_prediction_type]}%\n")
