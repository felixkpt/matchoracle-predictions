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

def prevent_equals_in_ft(prediction):
    ft_hda_pick = int(prediction['ft_hda_pick'])
    ft_home_win_proba = prediction['ft_home_win_proba']
    ft_draw_proba = prediction['ft_draw_proba']
    ft_away_win_proba = prediction['ft_away_win_proba']

    # Normalize to sum up to 100
    totals = ft_home_win_proba + ft_draw_proba + ft_away_win_proba
    ft_home_win_proba = round(ft_home_win_proba / totals * 100)
    ft_draw_proba = round(ft_draw_proba / totals * 100)
    ft_away_win_proba = round(ft_away_win_proba / totals * 100)

    # Check if draw probability is greater than 0 then we can do some adjustments on HDA
    if ft_draw_proba > 0:
        if ft_home_win_proba == ft_draw_proba and ft_draw_proba == ft_away_win_proba or ft_home_win_proba == ft_away_win_proba:
            # Adjusting probabilities and HDA values
            ft_home_win_proba -= 1
            ft_draw_proba += 2
            ft_away_win_proba -= 1
        elif ft_home_win_proba == ft_draw_proba:
            # Adjusting probabilities and HDA values
            ft_home_win_proba -= 1
            ft_draw_proba += 1
        elif ft_draw_proba == ft_away_win_proba:
            # Adjusting probabilities and HDA values
            ft_away_win_proba -= 1
            ft_draw_proba += 1

    ft_hda_pick = 0
    if ft_draw_proba > ft_home_win_proba and ft_draw_proba > ft_away_win_proba:
        ft_hda_pick = 1
    if ft_away_win_proba > ft_home_win_proba and ft_away_win_proba > ft_draw_proba:
        ft_hda_pick = 2


    prediction['ft_hda_pick'] = ft_hda_pick
    prediction['ft_home_win_proba'] = ft_home_win_proba
    prediction['ft_draw_proba'] = ft_draw_proba
    prediction['ft_away_win_proba'] = ft_away_win_proba

    return prediction


def prevent_equals_in_ht(prediction):
    if not prediction['ht_hda_pick']:
        return prediction

    ht_hda_pick = int(prediction['ht_hda_pick'])
    ht_home_win_proba = prediction['ht_home_win_proba']
    ht_draw_proba = prediction['ht_draw_proba']
    ht_away_win_proba = prediction['ht_away_win_proba']

    # Normalize to sum up to 100
    totals = ht_home_win_proba + ht_draw_proba + ht_away_win_proba
    ht_home_win_proba = int(ht_home_win_proba / totals * 100)
    ht_draw_proba = int(ht_draw_proba / totals * 100)
    ht_away_win_proba = int(ht_away_win_proba / totals * 100)

    # Check if draw probability is greater than 0 then we can do some adjustments on HDA
    if ht_draw_proba > 0:
        if ht_home_win_proba == ht_draw_proba and ht_draw_proba == ht_away_win_proba or ht_home_win_proba == ht_away_win_proba:
            # Adjusting probabilities and HDA values
            ht_home_win_proba -= 1
            ht_draw_proba += 2
            ht_away_win_proba -= 1
        elif ht_home_win_proba == ht_draw_proba:
            # Adjusting probabilities and HDA values
            ht_home_win_proba -= 1
            ht_draw_proba += 1
        elif ht_draw_proba == ht_away_win_proba:
            # Adjusting probabilities and HDA values
            ht_away_win_proba -= 1
            ht_draw_proba += 1

    # Re evaluate pick
    ht_hda_pick = 0
    if ht_draw_proba > ht_home_win_proba and ht_draw_proba > ht_away_win_proba:
        ht_hda_pick = 1
    if ht_away_win_proba > ht_home_win_proba and ht_away_win_proba > ht_draw_proba:
        ht_hda_pick = 2

    prediction['ht_hda_pick'] = ht_hda_pick
    prediction['ht_home_win_proba'] = ht_home_win_proba
    prediction['ht_draw_proba'] = ht_draw_proba
    prediction['ht_away_win_proba'] = ht_away_win_proba

    return prediction
