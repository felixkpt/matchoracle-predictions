import numpy as np


def normalizer(predict_proba):
    pred_proba_values = np.array(predict_proba)

    # Convert probabilities to percentages
    percentages = pred_proba_values * 100

    # Round percentages and organize into a list of lists
    normalized_preds = []
    for percentage in percentages:
        rounded_percentage = [round(p) for p in percentage]
        if rounded_percentage[0] > 0:
            normalized_preds.append(rounded_percentage)

    return normalized_preds


def prevent_equals_in_overs(prediction):
    over_under15_pick = int(prediction['over_under15_pick'])
    over15_proba = prediction['over15_proba']
    under15_proba = prediction['under15_proba']

    over_under25_pick = int(prediction['over_under25_pick'])
    over25_proba = prediction['over25_proba']
    under25_proba = prediction['under25_proba']

    over_under35_pick = int(prediction['over_under35_pick'])
    over35_proba = prediction['over35_proba']
    under35_proba = prediction['under35_proba']

    # Special over boost in cases where winner is, but slight
    ft_draw_proba = prediction['ft_draw_proba']
    gg_proba = prediction['gg_proba']
    if ft_draw_proba <= 34 and over25_proba >= 45 and gg_proba >= 58:
        over25_proba = 51
        under25_proba = 49

    # Handle case where over25_proba is lagging
    if over15_proba > over25_proba and over35_proba > over25_proba:
        over25_proba = int((over15_proba + over35_proba) / 2)
        under25_proba = 100 - over25_proba

    # fix overs normal behavior
    if over35_proba >= over25_proba:
        over25_proba = over35_proba + 5
        under25_proba = 100 - over25_proba

    if over25_proba >= over15_proba:
        over15_proba = over25_proba + 5
        under15_proba = 100 - over15_proba

    # Normalize to sum up to 100
    totals = over15_proba + under15_proba
    over15_proba = round(over15_proba / totals * 100)
    under15_proba = round(under15_proba / totals * 100)

    totals = over25_proba + under25_proba
    over25_proba = round(over25_proba / totals * 100)
    under25_proba = round(under25_proba / totals * 100)

    totals = over35_proba + under35_proba
    over35_proba = round(over35_proba / totals * 100)
    under35_proba = round(under35_proba / totals * 100)

    # fix unbalanced issue
    if over15_proba == 50 or under15_proba == 50:
        if over25_proba >= 50:
            over15_proba = 51
            under15_proba = 49
        else:
            over15_proba = 49
            under15_proba = 51

    if over25_proba == 50 or under25_proba == 50:
        if over35_proba >= 50:
            over25_proba = 51
            under25_proba = 49
        else:
            over25_proba = 49
            under25_proba = 51

    if over35_proba == 50 or under35_proba == 50:
        if over25_proba >= 60:
            over35_proba = 51
            under35_proba = 49
        else:
            over35_proba = 49
            under35_proba = 51

    # Re evaluate picks
    over_under15_pick = 0
    if over15_proba > under15_proba:
        over_under15_pick = 1

    over_under25_pick = 0
    if over25_proba > under25_proba:
        over_under25_pick = 1

    over_under35_pick = 0
    if over35_proba > under35_proba:
        over_under35_pick = 1

    prediction['over15_proba'] = over15_proba
    prediction['under15_proba'] = under15_proba
    prediction['over_under15_pick'] = over_under15_pick

    prediction['over25_proba'] = over25_proba
    prediction['under25_proba'] = under25_proba
    prediction['over_under25_pick'] = over_under25_pick

    prediction['over35_proba'] = over35_proba
    prediction['under35_proba'] = under35_proba
    prediction['over_under35_pick'] = over_under35_pick

    return prediction
