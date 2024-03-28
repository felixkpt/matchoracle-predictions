# Importing necessary modules
from app.helpers.scores import scores
from app.train_predictions.hyperparameters.hyperparameters import get_occurrences

# Function to normalize predictions


def predictions_normalizer(prediction, compe_data):
    # Get occurrences from hyperparameters
    occurrences = get_occurrences(compe_data, 'cs_target', 0.01)

    # Generating a list of dictionaries containing scores
    scores_dict = scores(occurrences)

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

    # Determining margin categories for full-time HDA, both teams to score (BTS), and over 3.5 goals
    # marion mar

    ft_home_margin = ft_draw_margin = ft_away_margin = 0

    print('Normalizer: ', ft_hda_pick, ft_home_win_proba, ft_draw_proba, ft_away_win_proba, '<<<< HDA')
    if ft_home_win_proba > ft_draw_proba and ft_home_win_proba > ft_away_win_proba:
        ft_home_margin = 1 if ft_home_win_proba < 34 else 3 if ft_home_win_proba < 40 else 3 if ft_home_win_proba < 55 else 4
    elif ft_draw_proba > ft_home_win_proba and ft_draw_proba > ft_away_win_proba:
        ft_draw_margin = 1 if ft_draw_proba < 34 else 2 if ft_draw_proba < 40 else 2 if ft_draw_proba < 55 else 3
    elif ft_away_win_proba > ft_home_win_proba and ft_away_win_proba > ft_draw_proba:
        ft_away_margin = 1 if ft_away_win_proba < 34 else 2 if ft_away_win_proba < 40 else 3 if ft_away_win_proba < 55 else 4

    # Adjusting probabilities and half-time HDA values if they are equal
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

    bts_pick = int(prediction['bts_pick'])
    gg_proba = prediction['gg_proba']
    bts_margin = 0 if gg_proba < 49 else 1 if gg_proba < 65 else 2 if gg_proba < 85 else 3

    over15_pick = int(prediction['over15_pick'])

    over25_pick = int(prediction['over25_pick'])

    over35_pick = int(prediction['over35_pick'])
    over35_margin = 0 if over35_pick < 40 else 1 if over35_pick < 55 else 2 if over35_pick < 60 else 3

    # Filtering scores based on full-time HDA
    scores_dict = [
        score for score in scores_dict if score["hda"] == ft_hda_pick]

    scores_dict = [
        score for score in scores_dict if score["home_margin"] <= ft_home_margin]

    # Filtering scores based on full-time away margin
    scores_dict = [
        score for score in scores_dict if score["away_margin"] <= ft_away_margin]

    # Filtering scores based on both teams to score margin
    scores_dict = [
        score for score in scores_dict if score["bts_margin"] <= bts_margin]

    # Filtering scores based on over 3.5 goals margin
    scores_dict = [
        score for score in scores_dict if score["over35_margin"] <= over35_margin]

    # Extracting prediction values for comparison
    prediction_values = [ft_hda_pick, ft_home_margin, ft_draw_margin,
                         ft_away_margin, bts_pick, bts_margin, over15_pick, over25_pick, over35_pick]

    # Initializing votes for each score
    votes = [0] * len(scores_dict)

    # Counting votes for each score based on prediction values
    for i, score in enumerate(scores_dict):
        for j, key in enumerate(['hda', 'home_margin', 'draw_margin', 'away_margin', 'bts', 'bts_margin', 'over15', 'over25', 'over35']):
            votes[i] += 1 if prediction_values[j] == score[key] else 0

    cs = 0
    if len(votes) > 0 or True:
        # Finding the maximum number of votes
        max_votes = max(votes)

        # Finding the indices of scores with the maximum number of votes
        best_match_indices = [
            i for i, vote in enumerate(votes) if vote == max_votes]

        # Prefer the one with the highest index in case of a tie
        best_match_index = max(best_match_indices)
        best_match = scores_dict[best_match_index]

        # Assigning the best match's correct score to the prediction
        cs = best_match['number']

    print('CS RES:', cs)
    prediction['cs'] = cs

    return prediction
