def filter_scores_dict(scores_dict, prediction, bts_margin, ft_hda_pick, over25_proba, gg_proba, ft_home_margin, ft_away_margin):
    over35_proba = prediction['over35_proba']

    # Determine the gmargin category for over 3.5 goals
    over35_margin = 0 if over35_proba <= 40 else 1 if over35_proba <= 55 else 2 if over35_proba <= 60 else 3

    ft_home_win_proba = prediction['ft_home_win_proba']
    ft_draw_proba = prediction['ft_draw_proba']
    ft_away_win_proba = prediction['ft_away_win_proba']

    # Strict filter based on HDA
    if ft_home_win_proba < 50:
        scores_dict = [
            score for score in scores_dict if score["home_margin"] <= 3]
    elif ft_draw_proba < 50:
        scores_dict = [
            score for score in scores_dict if score["draw_margin"] <= 3]
    elif ft_away_win_proba < 50:
        scores_dict = [
            score for score in scores_dict if score["away_margin"] <= 3]

    # Strict filter based on HDA
    if ft_home_win_proba < 60:
        scores_dict = [
            score for score in scores_dict if score["home_margin"] <= 4]
    elif ft_draw_proba < 60:
        scores_dict = [
            score for score in scores_dict if score["draw_margin"] <= 4]
    elif ft_away_win_proba < 60:
        scores_dict = [
            score for score in scores_dict if score["away_margin"] <= 4]
    
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

    # Filtering scores based on over 2.5 goals predict, take overs
    if over25_proba >= 56:
        scores_dict = [
            score for score in scores_dict if score["over25"] == 1]
        # Fallback to previous state if no scores remain
        scores_dict = scores_dict if len(
            scores_dict) > 0 else original_scores_dict
        original_scores_dict = scores_dict.copy()

    # Filtering scores based on over 3.5 goals predict, take unders
    if over25_proba < 55 and over35_proba < 45 and gg_proba < 55:
        scores_dict = [
            score for score in scores_dict if score["over35"] == 0]
        # Fallback to previous state if no scores remain
        scores_dict = scores_dict if len(
            scores_dict) > 0 else original_scores_dict
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
    # print('SCORES DICT:::', scores_dict)

    # Filtering scores based on full-time away margin
    scores_dict = [
        score for score in scores_dict if score["away_margin"] <= ft_away_margin]
    # Fallback to previous state if no scores remain
    scores_dict = scores_dict if len(scores_dict) > 0 else original_scores_dict
    original_scores_dict = scores_dict.copy()

    return scores_dict
