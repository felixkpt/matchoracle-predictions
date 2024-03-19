import numpy as np

def normalizer(predict_proba):
    # Given decision function values
    pred_proba_values = np.array(predict_proba)

    # Calculate the probabilities by applying the softmax function
    probas = np.exp(pred_proba_values) / \
        np.sum(np.exp(pred_proba_values), axis=1, keepdims=True)

    # Convert probabilities to percentages
    percentages = probas * 100

    # Define the desired range for probabilities
    min_y_pred_0_prob = 10
    min_y_pred_1_prob = 10

    normalized_preds = []

    for i, percentage in enumerate(percentages):
        # Ensure draw probability is within the desired range
        if percentage[0] < min_y_pred_0_prob:
            less_by_prob = min_y_pred_0_prob - percentage[0]
            percentage[0] = min_y_pred_0_prob
            percentage[1] -= less_by_prob  # Redistribute from 1

        if percentage[1] < min_y_pred_1_prob:
            less_by_prob = min_y_pred_1_prob - percentage[1]
            percentage[1] = min_y_pred_1_prob
            percentage[0] -= less_by_prob  # Redistribute to 0

        y_pred_0_percentage = round(percentage[0])
        y_pred_1_percentage = round(percentage[1])

        match_data = [y_pred_0_percentage, y_pred_1_percentage]
        normalized_preds.append(match_data)

    return normalized_preds
