# Importing necessary modules
from app.helpers.scores import scores
from app.train_predictions.hyperparameters.hyperparameters import get_occurrences
from app.predictions_normalizers.hda_normalizer import prevent_equals_in_ft, prevent_equals_in_ht
from app.predictions_normalizers.over_normalizer import prevent_equals_in_overs
from app.predictions_normalizers.bts_normalizer import prevent_equals_in_bts
from app.predictions_normalizers.filter_scores_dict import filter_scores_dict

# Function to normalize predictions


def predictions_normalizer(cs_model_type, prediction, compe_data):
    # Get occurrences from hyperparameters
    occurrences = get_occurrences(cs_model_type, compe_data, 'cs_target', 0.1)

    # Generating a list of dictionaries containing scores
    scores_dict = scores(occurrences)

    # Adjusting probabilities and full-time HDA values if they are equal
    prediction = prevent_equals_in_ft(prediction)

    ft_hda_pick = int(prediction['ft_hda_pick'])
    ft_home_win_proba = prediction['ft_home_win_proba']
    ft_draw_proba = prediction['ft_draw_proba']
    ft_away_win_proba = prediction['ft_away_win_proba']

    # Printing debug information
    print('Normalized HDA >>> ', ft_hda_pick, ft_home_win_proba,
          ft_draw_proba, ft_away_win_proba)

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

    # Adjusting probabilities and full-time HDA values if they are equal
    prediction = prevent_equals_in_ht(prediction)
    if prediction['ht_hda_pick']:
        ht_hda_pick = int(prediction['ht_hda_pick'])
        ht_home_win_proba = prediction['ht_home_win_proba']
        ht_draw_proba = prediction['ht_draw_proba']
        ht_away_win_proba = prediction['ht_away_win_proba']

    # Normalize and extract over/under prediction values for comparison
    prediction = prevent_equals_in_overs(prediction)
    over_under15_pick = int(prediction['over_under15_pick'])
    over15_proba = prediction['over15_proba']
    under15_proba = prediction['under15_proba']

    over_under25_pick = int(prediction['over_under25_pick'])
    over25_proba = prediction['over25_proba']
    under25_proba = prediction['under25_proba']

    over_under35_pick = int(prediction['over_under35_pick'])
    over35_proba = prediction['over35_proba']
    under35_proba = prediction['under35_proba']

    print('overs probas', over15_proba, over25_proba, over35_proba)
    print('overs', over_under15_pick, over_under25_pick, over_under35_pick)

    # Normalize and determine the margin category for both teams to score (BTS)
    prediction = prevent_equals_in_bts(prediction)
    bts_pick = int(prediction['bts_pick'])
    gg_proba = prediction['gg_proba']
    ng_proba = prediction['ng_proba']

    bts_margin = 0
    if over25_proba >= 55:
        if gg_proba >= 70:
            bts_margin = 3
        elif gg_proba >= 60:
            bts_margin = 2
        elif gg_proba >= 50:
            bts_margin = 1
    else:
        if gg_proba >= 50:
            bts_margin = 1

    if ft_hda_pick == 1 and gg_proba > 45:
        bts_margin = bts_margin + 1

    # reduce high home / away margins in case of high bts margin
    if ft_home_margin >= 3 and bts_margin >= 2:
        ft_home_margin -= 1
        print('high ft_home_margin reduced due to high bts.')
    if ft_away_margin >= 3 and bts_margin >= 2:
        ft_away_margin -= 1
        print('high ft_away_margin reduced due to high bts.')

    print('gg / bts_margin', gg_proba, bts_margin)

    # Work on scores_dict
    scores_dict = filter_scores_dict(scores_dict, prediction, bts_margin,
                                     ft_hda_pick, over25_proba, gg_proba, ft_home_margin, ft_away_margin)

    # Extracting prediction values for comparison
    prediction_values = [ft_hda_pick, ft_home_margin, ft_draw_margin,
                         ft_away_margin, bts_pick, bts_margin, over_under25_pick, over_under35_pick]

    # Initializing votes for each score
    votes = [0] * len(scores_dict)

    # Counting votes for each score based on prediction values
    for i, score in enumerate(scores_dict):
        for j, key in enumerate(['hda', 'home_margin', 'draw_margin', 'away_margin', 'bts', 'bts_margin', 'over25', 'over35']):
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
        cs_echo = best_match['cs']

        # correct bts
        if best_match['bts'] != bts_pick:
            if bts_pick == 0 and gg_proba >= 47:
                gg_proba = 51
                ng_proba = 49
                bts_pick = 1
            if bts_pick == 0 and ng_proba >= 47:
                gg_proba = 49
                ng_proba = 51
                bts_pick = 0

            prediction['gg_proba'] = gg_proba
            prediction['ng_proba'] = ng_proba
            prediction['bts_pick'] = bts_pick
        print('Normalized gg', gg_proba)

        # over bts
        if best_match['bts_margin'] >= 2 and gg_proba >= 55 and over25_proba > 45 and over25_proba < 50:
            over25_proba = 52
            under25_proba = 48
            over_under25_pick = 1

            prediction['over25_proba'] = over25_proba
            prediction['under25_proba'] = under25_proba
            prediction['over_under25_pick'] = over_under25_pick
            print('Normalized overs probas favor over', over25_proba,
                  over25_proba, over35_proba)

        # overs only
        if best_match['over25'] == 0 and over25_proba >= 50 and over25_proba <= 53 and gg_proba <= 55:
            over25_proba = 49
            under25_proba = 51
            over_under25_pick = 0

            prediction['over25_proba'] = over25_proba
            prediction['under25_proba'] = under25_proba
            prediction['over_under25_pick'] = over_under25_pick
            print('Normalized over2 proba favor under', over25_proba)

        if best_match['over25'] == 1 and over25_proba <= 50 and (over25_proba >= 47 or (over25_proba >= 45 and gg_proba >= 50)):
            over25_proba = 51
            under25_proba = 49
            over_under25_pick = 1

            prediction['over25_proba'] = over25_proba
            prediction['under25_proba'] = under25_proba
            prediction['over_under25_pick'] = over_under25_pick
            print('Normalized over25 proba favor over25 b\'cause\'v CS', over25_proba)

    # Print the result of correct score
    print(f'CS RES: {cs} ({cs_echo})')
    prediction['cs'] = cs

    return prediction
