from app.helpers.scores import scores
from app.train_predictions.hyperparameters.hyperparameters import get_occurrences


def predictions_normalizer(prediction, compe_data):

    # Get occurrences from hyperparams
    occurences = get_occurrences(
        compe_data, 'cs_target')

    dictionary_list = scores(occurences, min_threshold=0.2)

    hda = int(prediction['hda'])
    home_win_proba = prediction['home_win_proba']
    draw_proba = prediction['draw_proba']
    away_win_proba = prediction['away_win_proba']
    if home_win_proba == draw_proba and draw_proba == away_win_proba:
        home_win_proba -= 1
        prediction['home_win_proba'] = home_win_proba
        draw_proba += 2
        prediction['draw_proba'] = draw_proba
        away_win_proba -= 1
        prediction['away_win_proba'] = away_win_proba
        hda = 1
        prediction['hda'] = hda

    home_margin = 0 if home_win_proba < 34 else 1 if home_win_proba < 40 else 2 if home_win_proba < 55 else 3
    draw_margin = 0 if draw_proba < 34 else 1 if draw_proba < 40 else 2 if draw_proba < 55 else 3
    away_margin = 0 if away_win_proba < 34 else 1 if away_win_proba < 40 else 2 if away_win_proba < 55 else 3
    bts = int(prediction['bts'])
    gg_proba = prediction['gg_proba']
    bts_margin = 0 if gg_proba < 49 else 1 if gg_proba < 65 else 2 if gg_proba < 85 else 3
    over15 = int(prediction['over15'])
    over25 = int(prediction['over25'])
    over35 = int(prediction['over35']) + 5
    over35_margin = 0 if over35 < 40 else 1 if over35 < 55 else 2 if over35 < 60 else 3

    # filter follow hda
    dictionary_list = [
        score for score in dictionary_list if score["hda"] == hda]
    
    # filter follow home_margin margin
    dictionary_list = [
        score for score in dictionary_list if score["home_margin"] <= home_margin]
    # filter follow away_margin margin
    dictionary_list = [
        score for score in dictionary_list if score["away_margin"] <= away_margin]
    # filter follow bts margin
    dictionary_list = [
        score for score in dictionary_list if score["bts_margin"] <= bts_margin]
    # filter follow over35 margin
    dictionary_list = [
        score for score in dictionary_list if score["over35_margin"] <= over35_margin]

    prediction_values = [
        hda,
        home_margin,
        draw_margin,
        away_margin,
        bts,
        bts_margin,
        over15,
        over25,
        over35,
    ]

    votes = [0] * len(dictionary_list)

    for i, score in enumerate(dictionary_list):
        for j, key in enumerate(['hda', 'home_margin', 'draw_margin', 'away_margin', 'bts', 'bts_margin', 'over15', 'over25', 'over35']):
            votes[i] += 1 if prediction_values[j] == score[key] else 0

    max_votes = max(votes)
    best_match_indices = [
        i for i, vote in enumerate(votes) if vote == max_votes]

    # Prefer the one with the highest index in case of a tie
    best_match_index = max(best_match_indices)
    best_match = dictionary_list[best_match_index]

    cs = best_match['cs']
    print('CS RES:', cs)

    prediction['cs'] = cs
    return prediction
