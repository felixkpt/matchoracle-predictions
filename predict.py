from configs.settings import API_BASE_URL
from app.predictions.ft_hda_predictions import ft_hda_predictions
from app.predictions.ht_hda_predictions import ht_hda_predictions
from app.predictions.bts_predictions import bts_predictions
from app.predictions.over15_predictions import over15_predictions
from app.predictions.over25_predictions import over25_predictions
from app.predictions.over35_predictions import over35_predictions
from app.predictions.cs_predictions import cs_predictions
from app.matches.load_matches import load_for_predictions
from configs.logger import Logger
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import requests
import argparse
from configs.active_competitions.competitions_data import get_trained_competitions
from app.predictions_normalizers.predictions_normalizer import predictions_normalizer

def predict(user_token):
    print("\n............... START PREDICTIONS ..................\n")

    PREDICTION_TYPE = f"regular_prediction_10_6_4"

    parser = argparse.ArgumentParser(
        description='Predictions with different configurations.')
    parser.add_argument('--competition', type=int, help='Competition ID')
    parser.add_argument('--target', choices=['hda', 'ft_hda', 'ht_hda', 'bts', 'over15', 'over25', 'over35', 'cs'],
                        help='Target for predictions')

    parser.add_argument('--ignore-filters', action='store_true',
                        help='Ignore filters data')
    parser.add_argument('--ignore-timimg', action='store_true',
                        help='Ignore timing data')

    args, extra_args = parser.parse_known_args()
    target = args.target
    ignore_filters = args.ignore_filters
    ignore_timimg = args.ignore_timimg

    print(f"Main Prediction Target: {target if target else 'all'}")
    print(f"")

    # If competition_id is provided, use it; otherwise, fetch from the backend API
    competition_ids = [
        args.competition] if args.competition is not None else get_trained_competitions(ignore_filters, ignore_timimg)

    # Loop over competition IDs
    for COMPETITION_ID in competition_ids:
        # Calculate from_date and to_date
        target_date_init = datetime.strptime('2024-01-01', '%Y-%m-%d')
        compe_data = {}
        compe_data['id'] = COMPETITION_ID
        compe_data['prediction_type'] = PREDICTION_TYPE

        Logger.info(f"Competition: {COMPETITION_ID}")
        Logger.info(f"Prediction type: {PREDICTION_TYPE}\n")

        for i in range(0, 90):

            target_date = target_date_init - relativedelta(days=-1 * i)
            target_date = target_date.strftime("%Y-%m-%d")

            Logger.info(f"Competition: {COMPETITION_ID}")
            Logger.info(f"Date: {target_date}\n")

            matches = load_for_predictions(
                COMPETITION_ID, target_date, user_token)

            total_matches = len(matches)

            if total_matches == 0:
                print('No matches to make predictions!')
            else:
                print(f'Predicting {total_matches} matches...')

                ft_hda_preds = ft_hda_predictions(matches, compe_data)
                ht_hda_preds = ht_hda_predictions(matches, compe_data)

                bts_preds = bts_predictions(matches, compe_data)

                over15_preds = over15_predictions(matches, compe_data)
                over25_preds = over25_predictions(matches, compe_data)
                over35_preds = over35_predictions(matches, compe_data)

                cs_preds = cs_predictions(matches, compe_data)

                merge_and_store_predictions(user_token, compe_data, target_date, matches, ft_hda_preds, ht_hda_preds,
                                            bts_preds, over15_preds, over25_preds, over35_preds, cs_preds)
            print(f"______________\n")

    print(f"\n....... END PREDICTIONS, Happy coding! ........")


def merge_and_store_predictions(user_token, compe_data, target_date, matches, ft_hda_preds, ht_hda_preds,
                                bts_preds, over15_preds, over25_preds, over35_preds, cs_preds):

    ft_hda_preds, ft_hda_preds_proba = ft_hda_preds
    ht_hda_preds, ht_hda_preds_proba = ht_hda_preds

    bts_preds, bts_preds_proba = bts_preds

    over15_preds, over15_preds_proba = over15_preds
    over25_preds, over25_preds_proba = over25_preds
    over35_preds, over35_preds_proba = over35_preds

    cs_preds, cs_preds_proba = cs_preds

    predictions = []
    for i, match in enumerate(matches):

        ft_hda = str(ft_hda_preds[i])
        hda_proba = ft_hda_preds_proba[i]
        ft_home_win_proba = hda_proba[0]
        ft_draw_proba = hda_proba[1]
        ft_away_win_proba = hda_proba[2]

        ht_hda = None
        hda_proba = None
        ht_home_win_proba = None
        ht_draw_proba = None
        ht_away_win_proba = None
        try:
            ht_hda = str(ht_hda_preds[i])
            hda_proba = ht_hda_preds_proba[i]
            ht_home_win_proba = hda_proba[0]
            ht_draw_proba = hda_proba[1]
            ht_away_win_proba = hda_proba[2]
        except:
            TypeError

        bts = str(bts_preds[i])
        bts_proba = bts_preds_proba[i]

        over15 = str(over15_preds[i])
        over15_proba = over15_preds_proba[i]

        over25 = str(over25_preds[i])
        over25_proba = over25_preds_proba[i]

        over35 = str(over35_preds[i])
        over35_proba = over35_preds_proba[i]

        key = cs_preds[i]
        cs = str(cs_preds[i])
        cs_proba = cs_preds_proba[i][key]

        pred_obj = {
            'id': match['id'],

            'ft_hda_pick': ft_hda,
            'ft_home_win_proba': ft_home_win_proba,
            'ft_draw_proba': ft_draw_proba,
            'ft_away_win_proba': ft_away_win_proba,

            'ht_hda_pick': ht_hda,
            'ht_home_win_proba': ht_home_win_proba,
            'ht_draw_proba': ht_draw_proba,
            'ht_away_win_proba': ht_away_win_proba,

            'bts_pick': bts,
            'gg_proba': bts_proba[1],
            'ng_proba': bts_proba[0],

            'over15_pick': over15,
            'over15_proba': over15_proba[1],
            'under15_proba': over15_proba[0],

            'over25_pick': over25,
            'over25_proba': over25_proba[1],
            'under25_proba': over25_proba[0],

            'over35_pick': over35,
            'over35_proba': over35_proba[1],
            'under35_proba': over35_proba[0],

            'cs_unsensored': cs,
            'cs_proba_unsensored': cs_proba,
        }
        
        pred_obj = predictions_normalizer(pred_obj, compe_data)
        predictions.append(pred_obj)

    data = {
        'version': '1.0',
        'type': compe_data['prediction_type'],
        'competition_id': compe_data['id'],
        'date': str(target_date),
        'predictions': predictions}

    message = storePredictions(data, user_token)
    print(message)


def storePredictions(data, user_token):

    # Now that you have the user token, you can use it for other API requests.
    url = f"{API_BASE_URL}/admin/predictions/from-python-app/store-predictions"

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    json_data = json.dumps(data)
    # Make a GET request with the headers
    response = requests.post(url, data=json_data, headers=headers)
    response.raise_for_status()

    # Parse the JSON response
    message = response.json()['message']

    return message
