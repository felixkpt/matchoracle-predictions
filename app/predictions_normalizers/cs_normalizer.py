import numpy as np


def normalizer(predict_proba):
    # Given decision function values
    pred_proba_values = np.array(predict_proba)

    # Calculate the probabilities by applying the softmax function
    probas = np.exp(pred_proba_values) / \
        np.sum(np.exp(pred_proba_values), axis=1, keepdims=True)

    # Convert probabilities to percentages
    percentages = probas * 100

    normalized_preds = []

    for i, percentage in enumerate(percentages):
        match_data = [round(p, 2) for p in percentage]
        normalized_preds.append(max(match_data))

    return normalized_preds
