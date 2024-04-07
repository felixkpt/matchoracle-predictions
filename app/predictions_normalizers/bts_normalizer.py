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


def prevent_equals_in_bts(prediction):
    bts_pick = int(prediction['bts_pick'])
    gg_proba = prediction['gg_proba']
    ng_proba = prediction['ng_proba']

    over25_proba = prediction['over25_proba']

    # Normalize to sum up to 100
    totals = gg_proba + ng_proba
    gg_proba = round(gg_proba / totals * 100)
    ng_proba = round(ng_proba / totals * 100)

    # Special over boost in cases where winner is, but slight
    ft_draw_proba = prediction['ft_draw_proba']
    gg_proba = prediction['gg_proba']
    if ft_draw_proba >= 30 and over25_proba >= 50 and gg_proba < 50:
        gg_proba = 51
        ng_proba = 49

    if gg_proba == 50 or ng_proba == 50:
        if over25_proba >= 50:
            gg_proba = 51
            ng_proba = 49
            bts_pick = 1
        else:
            gg_proba = 49
            ng_proba = 51
            bts_pick = 0

    # Re evaluate picks
    bts_pick = 0
    if gg_proba > ng_proba:
        bts_pick = 1

    prediction['gg_proba'] = gg_proba
    prediction['ng_proba'] = ng_proba
    prediction['bts_pick'] = bts_pick

    return prediction
