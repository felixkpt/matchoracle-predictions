import numpy as np

def normalizer(predict_proba):
    pred_proba_values = np.array(predict_proba)

    # Convert probabilities to percentages
    percentages = pred_proba_values * 100

    # Round percentages and organize into a list of lists
    normalized_preds = []
    for percentage in percentages:
        rounded_percentage = [round(p) for p in percentage]
        normalized_preds.append(rounded_percentage)

    return normalized_preds
