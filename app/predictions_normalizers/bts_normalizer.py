import numpy as np


def normalizer(predict_proba):
    # Define the desired range for probabilities
    min_y_pred_0_prob = 10
    min_y_pred_1_prob = 10

    normalized_preds = []

    for i, proba in enumerate(predict_proba):
        percentage = [int(proba[0] * 100), int(proba[1] * 100)]

        # Ensure draw probability is within the desired range
        if percentage[0] < min_y_pred_0_prob:
            less_by_prob = min_y_pred_0_prob - percentage[0]
            percentage[0] = min_y_pred_0_prob
            percentage[1] -= less_by_prob  # Redistribute from 1

        if percentage[1] < min_y_pred_1_prob:
            less_by_prob = min_y_pred_1_prob - percentage[1]
            percentage[1] = min_y_pred_1_prob
            percentage[0] -= less_by_prob  # Redistribute to 0

        y_pred_0_percentage = percentage[0]
        y_pred_1_percentage = percentage[1]

        match_data = [y_pred_0_percentage, y_pred_1_percentage]
        normalized_preds.append(match_data)

    return normalized_preds
