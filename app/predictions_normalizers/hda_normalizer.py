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
    min_y_pred_1_prob = 15
    min_y_pred_2_prob = 15

    max_y_pred_1_prob = 50

    normalized_preds = []

    for i, percentage in enumerate(percentages):
        # Ensure draw probability is within the desired range
        
        # Handle 0
        # if percentage[0] < min_y_pred_0_prob:
        #     less_by_prob = min_y_pred_0_prob - percentage[1]
        #     percentage[0] = min_y_pred_0_prob
        #     percentage[1] -= less_by_prob / 2  # Redistribute to 1
        #     percentage[2] -= less_by_prob / 2  # Redistribute to 2

        # # Handle 1
        # if percentage[1] < min_y_pred_1_prob:
        #     less_by_prob = min_y_pred_1_prob - percentage[0]
        #     percentage[1] = min_y_pred_1_prob
        #     percentage[0] -= less_by_prob / 2  # Redistribute from 0
        #     percentage[2] -= less_by_prob / 2  # Redistribute from 2
        # elif percentage[1] > max_y_pred_1_prob:
        #     excess_by_prob = percentage[1] - max_y_pred_1_prob
        #     percentage[1] = max_y_pred_1_prob
        #     percentage[0] += excess_by_prob / 2  # Redistribute to 0
        #     percentage[2] += excess_by_prob / 2  # Redistribute to 2
        
        # # Handle 2
        # if percentage[2] < min_y_pred_2_prob:
        #     less_by_prob = min_y_pred_2_prob - percentage[2]
        #     percentage[2] = min_y_pred_2_prob
        #     percentage[0] -= less_by_prob / 2  # Redistribute to 1
        #     percentage[1] -= less_by_prob / 2  # Redistribute to 2

        y_pred_0_percentage = round(percentage[0])
        y_pred_1_percentage = round(percentage[1])
        y_pred_2_percentage = round(percentage[2])

        match_data = [y_pred_0_percentage, y_pred_1_percentage, y_pred_2_percentage]
        normalized_preds.append(match_data)

    return normalized_preds
