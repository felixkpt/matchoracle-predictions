import numpy as np
from app.helpers.functions import bound_probabilities


def normalizer(predict_proba):
    pred_proba_values = np.array(predict_proba)

    # Convert probabilities to percentages
    percentages = pred_proba_values * 100

    # Normalize and prevent hard 100/0 extremes
    normalized_preds = []
    for percentage in percentages:
        adjusted = bound_probabilities([round(p) for p in percentage], 7)
        normalized_preds.append(adjusted)

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
