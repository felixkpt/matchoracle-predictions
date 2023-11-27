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
            # print(
            #     f"Evaluating comparison for Prediction Type {prediction_type}...\n")

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

        aggregated_average_scores = []
        # Actual comparison of prediction_types (aka models)
        for prediction_type, competition_metrics in comparison_results.items():
            print(
                f"\nComparison results for Prediction Type {prediction_type}:\n")

            counts = 0
            average_score_totals = 0
            aggregated_average_score = 0
            for competition_id, metrics_data in competition_metrics.items():
                counts += 1
                print(f"Competition {competition_id} Metrics:")
                print(f"  Accuracy Score: {metrics_data['accuracy_score']}%")
                print(f"  Precision Score: {metrics_data['precision_score']}%")
                print(f"  F1 Score: {metrics_data['f1_score']}%")
                print(f"  Average Score: {metrics_data['average_score']}%\n")
                average_score_totals += metrics_data['average_score']

            aggregated_average_score = round(average_score_totals / counts, 2)
            # Store competition_metrics in the overall aggregated_average_score dictionary
            aggregated_average_scores.append({
                'prediction_type': prediction_type,
                'aggregated_average_score': aggregated_average_score
            })

            print(f'Aggregated average score: {aggregated_average_score}%')
            print('__________\n')

        return aggregated_average_scores

    def choose_best_model(self, comparison_results):
        print(f"Choosing the best model for Target {self.target}:\n")

        # Dictionary to store the total average score for each prediction type
        total_average_scores = {}
        for dictionary in comparison_results:
            
            prediction_type = dictionary['prediction_type']
            aggregated_average_score = dictionary['aggregated_average_score']
            
            print(
                f"Total average score for Prediction Type {prediction_type} is {aggregated_average_score}%\n")
            # Store the total average score in the dictionary
            total_average_scores[prediction_type] = aggregated_average_score

        # Find the prediction type with the highest total average score
        best_prediction_type = max(
            total_average_scores, key=total_average_scores.get)
        # Sort the dictionary by keys
        sorted_hyperparameters = dict(
            sorted(total_average_scores.items(), key=lambda x: x[1], reverse=True))

        # Print the chosen best prediction type
        print(
            f"\nBest model for Target {self.target}:\n")
        print(f"Sorted hyperparameters: {sorted_hyperparameters}\n")
        print(f"Chosen Prediction Type: {best_prediction_type}")
        print(
            f"Total Average Score: {total_average_scores[best_prediction_type]}%\n")
