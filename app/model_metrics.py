import os
import json


class ModelMetrics:
    def __init__(self, target, prediction_types):
        self.target = target
        self.base_path = f"/home/felix/Documents/DEV/PY/matchoracle-predictions-v2/app/train_predictions/hyperparameters/"
        self.prediction_types = prediction_types
        self.metrics = self.load_metrics()

    def load_metrics(self):
        metrics = {}
        for prediction_type in self.prediction_types:
            json_path = os.path.join(
                self.base_path, f'{prediction_type}/{self.target}_hyperparams.json')
            with open(json_path, 'r') as file:
                metrics[prediction_type] = json.load(file)
        return metrics

    def compare_models(self):
        print(f"Comparison for Target {self.target}:\n")

        # Dictionary to store comparison results for each prediction type
        comparison_results = {}

        for prediction_type in self.prediction_types:
            print(f"Comparison for Prediction Type {prediction_type}:\n")

            competition_ids = list(self.metrics[prediction_type].keys())

            # Dictionary to store metrics for each competition
            competition_metrics = {}

            for competition_id in competition_ids:
                metrics_data = self.metrics[prediction_type][competition_id]

                # Retrieve metrics from current competition
                accuracy_score = metrics_data['accuracy_score']
                precision_score = metrics_data['precision_score']
                f1_score = metrics_data['f1_score']
                average_score = metrics_data['average_score']

                # Store metrics in the dictionary for later comparison
                competition_metrics[competition_id] = {
                    'accuracy_score': accuracy_score,
                    'precision_score': precision_score,
                    'f1_score': f1_score,
                    'average_score': average_score
                }

            # Store competition_metrics in the overall comparison_results dictionary
            comparison_results[prediction_type] = competition_metrics

        # Actual comparison of prediction_types (aka models)
        for prediction_type, competition_metrics in comparison_results.items():
            print(
                f"\nComparison results for Prediction Type {prediction_type}:\n")
            for competition_id, metrics_data in competition_metrics.items():
                print(f"Competition {competition_id} Metrics:")
                print(f"  Accuracy Score: {metrics_data['accuracy_score']}%")
                print(f"  Precision Score: {metrics_data['precision_score']}%")
                print(f"  F1 Score: {metrics_data['f1_score']}%")
                print(f"  Average Score: {metrics_data['average_score']}%\n")

        return comparison_results

    def choose_best_model(self, comparison_results):
        print(f"Choosing the best model for Target {self.target}:\n")

        # Dictionary to store the total average score for each prediction type
        total_average_scores = {}

        for prediction_type, competition_metrics in comparison_results.items():

            # Sum up the average scores for all competitions of the current prediction type
            total_average_score = sum(
                metrics_data['average_score'] for metrics_data in competition_metrics.values()
            )

            avg = int(total_average_score / len(competition_metrics))
            print(
                f"Calculated total average score for Prediction Type {prediction_type} is {avg}%\n")
            # Store the total average score in the dictionary
            total_average_scores[prediction_type] = avg

        # Find the prediction type with the highest total average score
        best_prediction_type = max(
            total_average_scores, key=total_average_scores.get)

        # Print the chosen best prediction type
        print(
            f"\nBest model for Target {self.target}:\n")
        print(f"Chosen Prediction Type: {best_prediction_type}")
        print(
            f"Total Average Score: {total_average_scores[best_prediction_type]}%\n")
