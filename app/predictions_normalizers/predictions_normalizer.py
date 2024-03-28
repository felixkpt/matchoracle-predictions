# Importing necessary modules
from app.helpers.scores import scores
from app.train_predictions.hyperparameters.hyperparameters import get_occurrences

# Function to normalize predictions


def predictions_normalizer(prediction, compe_data):
    # Get occurrences from hyperparameters
    occurrences = get_occurrences(compe_data, 'cs_target', 1.1)

    # Generating a list of dictionaries containing scores
    dictionary_list = scores(occurrences)

    print(dictionary_list)
    print('33')

    # Adjusting probabilities and full-time HDA values if they are equal
    ft_hda_pick = int(prediction['ft_hda_pick'])
    ft_home_win_proba = prediction['ft_home_win_proba']
    ft_draw_proba = prediction['ft_draw_proba']
    ft_away_win_proba = prediction['ft_away_win_proba']
    if ft_draw_proba > 0 and ft_home_win_proba == ft_draw_proba and ft_draw_proba == ft_away_win_proba:
        ft_home_win_proba -= 1
        prediction['ft_home_win_proba'] = ft_home_win_proba
        ft_draw_proba += 2
        prediction['ft_draw_proba'] = ft_draw_proba
        ft_away_win_proba -= 1
        prediction['ft_away_win_proba'] = ft_away_win_proba
        ft_hda_pick = 1
        prediction['ft_hda_pick'] = ft_hda_pick

    # Adjusting probabilities and full-time HDA values if they are equal
    if prediction['ht_hda_pick']:
        ht_hda_pick = int(prediction['ht_hda_pick'])
        ht_home_win_proba = prediction['ht_home_win_proba']
        ht_draw_proba = prediction['ht_draw_proba']
        ht_away_win_proba = prediction['ht_away_win_proba']
        if ht_draw_proba > 0 and ht_home_win_proba == ht_draw_proba and ht_draw_proba == ht_away_win_proba:
            ht_home_win_proba -= 1
            prediction['ht_home_win_proba'] = ht_home_win_proba
            ht_draw_proba += 2
            prediction['ht_draw_proba'] = ht_draw_proba
            ht_away_win_proba -= 1
            prediction['ht_away_win_proba'] = ht_away_win_proba
            ht_hda_pick = 1
            prediction['ht_hda_pick'] = ht_hda_pick

    # Determining margin categories for full-time HDA, both teams to score (BTS), and over 3.5 goals
    ft_home_margin = 0 if ft_home_win_proba < 34 else 1 if ft_home_win_proba < 40 else 2 if ft_home_win_proba < 55 else 3
    ft_draw_margin = 0 if ft_draw_proba < 34 else 1 if ft_draw_proba < 40 else 2 if ft_draw_proba < 55 else 3
    ft_away_margin = 0 if ft_away_win_proba < 34 else 1 if ft_away_win_proba < 40 else 2 if ft_away_win_proba < 55 else 3
    bts_pick = int(prediction['bts_pick'])
    gg_proba = prediction['gg_proba']
    bts_margin = 0 if gg_proba < 49 else 1 if gg_proba < 65 else 2 if gg_proba < 85 else 3
    over15_pick = int(prediction['over15_pick'])
    over25_pick = int(prediction['over25_pick'])
    over35_pick = int(prediction['over35_pick'])
    over35_margin = 0 if over35_pick < 40 else 1 if over35_pick < 55 else 2 if over35_pick < 60 else 3

    # Filtering scores based on full-time HDA
    dictionary_list = [
        score for score in dictionary_list if score["ft_hda_pick"] == ft_hda_pick]

    # Filtering scores based on full-time home margin
    dictionary_list = [
        score for score in dictionary_list if score["home_margin"] <= ft_home_margin]

    # Filtering scores based on full-time away margin
    dictionary_list = [
        score for score in dictionary_list if score["away_margin"] <= ft_away_margin]

    # Filtering scores based on both teams to score margin
    dictionary_list = [
        score for score in dictionary_list if score["bts_margin"] <= bts_margin]

    # Filtering scores based on over 3.5 goals margin
    dictionary_list = [
        score for score in dictionary_list if score["over35_margin"] <= over35_margin]

    
    # Extracting prediction values for comparison
    prediction_values = [ft_hda_pick, ft_home_margin, ft_draw_margin,
                         ft_away_margin, bts_pick, bts_margin, over15_pick, over25_pick, over35_pick]

    # Initializing votes for each score
    votes = [0] * len(dictionary_list)

    # Counting votes for each score based on prediction values
    for i, score in enumerate(dictionary_list):
        for j, key in enumerate(['ft_hda_pick', 'home_margin', 'draw_margin', 'away_margin', 'bts', 'bts_margin', 'over15', 'over25', 'over35']):
            votes[i] += 1 if prediction_values[j] == score[key] else 0

    # Finding the maximum number of votes
    max_votes = max(votes)

    # Finding the indices of scores with the maximum number of votes
    best_match_indices = [
        i for i, vote in enumerate(votes) if vote == max_votes]

    # Prefer the one with the highest index in case of a tie
    best_match_index = max(best_match_indices)
    best_match = dictionary_list[best_match_index]

    # Assigning the best match's correct score to the prediction
    cs = best_match['cs']
    print('CS RES:', cs)
    prediction['cs'] = cs

    return prediction
