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

    # Check if draw probability is greater than 0
    if ft_draw_proba > 0:
        if ft_home_win_proba == ft_draw_proba and ft_draw_proba == ft_away_win_proba or ft_home_win_proba == ft_away_win_proba:
            # Adjusting probabilities and HDA values
            ft_home_win_proba -= 1
            prediction['ft_home_win_proba'] = ft_home_win_proba
            ft_draw_proba += 2
            prediction['ft_draw_proba'] = ft_draw_proba
            ft_away_win_proba -= 1
            prediction['ft_away_win_proba'] = ft_away_win_proba
            ft_hda_pick = 1
            prediction['ft_hda_pick'] = ft_hda_pick
        elif ft_home_win_proba == ft_draw_proba:
            # Adjusting probabilities and HDA values
            ft_home_win_proba -= 1
            prediction['ft_home_win_proba'] = ft_home_win_proba
            ft_draw_proba += 1
            prediction['ft_draw_proba'] = ft_draw_proba
            ft_hda_pick = 1
            prediction['ft_hda_pick'] = ft_hda_pick
        elif ft_draw_proba == ft_away_win_proba:
            # Adjusting probabilities and HDA values
            ft_away_win_proba -= 1
            prediction['ft_away_win_proba'] = ft_away_win_proba
            ft_draw_proba += 1
            prediction['ft_draw_proba'] = ft_draw_proba
            ft_hda_pick = 1
            prediction['ft_hda_pick'] = ft_hda_pick

    # Printing debug information
    print('Normalizer: ', ft_hda_pick, ft_home_win_proba,
          ft_draw_proba, ft_away_win_proba, '<<<< HDA')

    # Determining margin categories for full-time HDA, both teams to score (BTS), and over 3.5 goals
    ft_home_margin = ft_draw_margin = ft_away_margin = 0

    # Determining margin categories based on probabilities
    m1_cutoff = 40
    m2_cutoff = 55
    m3_cutoff = 73
    if ft_home_win_proba > ft_draw_proba and ft_home_win_proba > ft_away_win_proba:
        ft_home_margin = 1 if ft_home_win_proba <= m1_cutoff else 2 if ft_home_win_proba <= m2_cutoff else 3 if ft_home_win_proba <= m3_cutoff else 4
    elif ft_draw_proba > ft_home_win_proba and ft_draw_proba > ft_away_win_proba:
        ft_draw_margin = 1 if ft_draw_proba <= m1_cutoff else 2 if ft_draw_proba <= m2_cutoff else 3 if ft_draw_proba <= m3_cutoff else 4
    elif ft_away_win_proba > ft_home_win_proba and ft_away_win_proba > ft_draw_proba:
        ft_away_margin = 1 if ft_away_win_proba <= m1_cutoff else 2 if ft_away_win_proba <= m2_cutoff else 3 if ft_away_win_proba <= m3_cutoff else 4

    # Adjusting probabilities and half-time HDA values if they are equal
    if prediction['ht_hda_pick']:
        ht_hda_pick = int(prediction['ht_hda_pick'])
        ht_home_win_proba = prediction['ht_home_win_proba']
        ht_draw_proba = prediction['ht_draw_proba']
        ht_away_win_proba = prediction['ht_away_win_proba']
        if ht_draw_proba > 0:
            if ht_home_win_proba == ht_draw_proba and ht_draw_proba == ht_away_win_proba or ht_hda_pick == ht_away_win_proba:
                ht_home_win_proba -= 1
                prediction['ht_home_win_proba'] = ht_home_win_proba
                ht_draw_proba += 2
                prediction['ht_draw_proba'] = ht_draw_proba
                ht_away_win_proba -= 1
                prediction['ht_away_win_proba'] = ht_away_win_proba
                ht_hda_pick = 1
                prediction['ht_hda_pick'] = ht_hda_pick
            elif ht_home_win_proba == ht_draw_proba:
                ht_home_win_proba -= 1
                prediction['ht_home_win_proba'] = ht_home_win_proba
                ht_draw_proba += 1
                prediction['ht_draw_proba'] = ht_draw_proba
                ht_hda_pick = 1
                prediction['ht_hda_pick'] = ht_hda_pick
            elif ht_draw_proba == ht_away_win_proba:
                ht_away_win_proba -= 1
                prediction['ht_away_win_proba'] = ht_away_win_proba
                ht_draw_proba += 1
                prediction['ht_draw_proba'] = ht_draw_proba
                ht_hda_pick = 1
                prediction['ht_hda_pick'] = ht_hda_pick

    # Determine the margin category for both teams to score (BTS)
    bts_pick = int(prediction['bts_pick'])
    gg_proba = prediction['gg_proba']
    bts_margin = 0 if gg_proba <= 49 else 1 if gg_proba <= 60 else 2 if gg_proba < 85 else 3

    # Extracting prediction values for comparison
    over15_pick = int(prediction['over15_pick'])
    over15_proba = prediction['over15_proba']

    over25_pick = int(prediction['over25_pick'])
    over25_proba = prediction['over25_proba']

    over35_pick = int(prediction['over35_pick'])
    over35_proba = prediction['over35_proba']

    print('overs', over15_pick, over25_pick, over35_pick)

    # Determine the margin category for over 3.5 goals
    over35_margin = 0 if over35_proba <= 40 else 1 if over35_proba <= 55 else 2 if over35_proba <= 60 else 3

    # Make a copy of the original scores_dict
    original_scores_dict = scores_dict.copy()

    # Filtering scores based on both teams to score margin
    scores_dict = [
        score for score in scores_dict if (bts_margin == score["bts_margin"] or score["bts_margin"] > 0 and score["bts_margin"] <= bts_margin)]
    # Fallback to previous state if no scores remain
    scores_dict = scores_dict if len(scores_dict) > 0 else original_scores_dict
    original_scores_dict = scores_dict.copy()

    # Filtering scores based on full-time HDA
    scores_dict = [
        score for score in scores_dict if score["hda"] == ft_hda_pick]
    # Fallback to previous state if no scores remain
    scores_dict = scores_dict if len(scores_dict) > 0 else original_scores_dict
    original_scores_dict = scores_dict.copy()

   # Filtering scores based on over 3.5 goals margin
    scores_dict = [
        score for score in scores_dict if score["over35_margin"] <= over35_margin + 2]
    # Fallback to previous state if no scores remain
    scores_dict = scores_dict if len(scores_dict) > 0 else original_scores_dict
    original_scores_dict = scores_dict.copy()

    # for s in original_scores_dict:
    #     print(s['cs'])
    # print(f'----')
    # print('applying:', over35_proba, over35_margin)
    # for s in scores_dict:
    #     print(s['cs'])

    # Filtering scores based on home margin
    scores_dict = [
        score for score in scores_dict if score["home_margin"] <= ft_home_margin]
    # Fallback to previous state if no scores remain
    scores_dict = scores_dict if len(scores_dict) > 0 else original_scores_dict
    original_scores_dict = scores_dict.copy()

    scores_dict = [
        score for score in scores_dict if score["home_margin"] == ft_home_margin]
    # Fallback to previous state if no scores remain
    scores_dict = scores_dict if len(scores_dict) > 0 else original_scores_dict
    original_scores_dict = scores_dict.copy()

    # Filtering scores based on full-time away margin
    scores_dict = [
        score for score in scores_dict if score["away_margin"] <= ft_away_margin]
    # Fallback to previous state if no scores remain
    scores_dict = scores_dict if len(scores_dict) > 0 else original_scores_dict
    original_scores_dict = scores_dict.copy()

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

    # Print the result of correct score
    print('CS RES:', cs)
    prediction['cs'] = cs

    return prediction
