from configs.settings import API_BASE_URL
from app.predictions.hda_predictions import hda_predictions
from app.predictions.bts_predictions import bts_predictions
from app.predictions.over25_predictions import over25_predictions
from app.predictions.cs_predictions import cs_predictions
from app.matches.load_matches import load_for_predictions
from configs.logger import Logger
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import json

# Define constants
# 48, 47, 148
COMPETITION_ID = 48


def predict(user_token):
    print("\n............... START PREDICTIONS ..................\n")

    # Calculate from_date and to_date
    target_date_init = datetime.strptime('2023-09-01', '%Y-%m-%d')

    for i in range(0, 70):

        target_date = target_date_init - relativedelta(days=-1 * i)
        target_date = target_date.strftime("%Y-%m-%d")

        Logger.info(f"Competition: {COMPETITION_ID}")
        Logger.info(f"Date: {target_date}\n")

        matches = load_for_predictions(COMPETITION_ID, target_date, user_token)

        total_matches = len(matches)

        if total_matches == 0:
            print('No matches to make predictions!')
            continue

        hda_preds = hda_predictions(matches, COMPETITION_ID)

        # below depends on hda_preds
        bts_preds = bts_predictions(matches, COMPETITION_ID)

        # below depends on hda_preds, bts_preds
        over25_preds = over25_predictions(matches, COMPETITION_ID)

        # below depends on hda_preds, bts_preds, over25_preds
        cs_preds = cs_predictions(matches, COMPETITION_ID)

        merge_and_store_predictions(user_token, target_date, matches, hda_preds,
                                    bts_preds, over25_preds, cs_preds)
        print(f"______________\n")

    print(f"\n....... END PREDICTIONS, Happy coding! ........")


def merge_and_store_predictions(user_token, target_date, matches, hda_preds, bts_preds, over25_preds, cs_preds):

    hda_preds, hda_preds_proba = hda_preds
    bts_preds, bts_preds_proba = bts_preds
    over25_preds, over25_preds_proba = over25_preds
    cs_preds, cs_preds_proba = cs_preds

    predictions = []
    for i, match in enumerate(matches):

        hda = str(hda_preds[i])
        hda_proba = hda_preds_proba[i]
        home_win_proba = hda_proba[1]
        draw_proba = hda_proba[0]
        away_win_proba = hda_proba[2]

        bts = str(bts_preds[i])
        bts_proba = bts_preds_proba[i]

        over25 = str(over25_preds[i])
        over25_proba = over25_preds_proba[i]

        key = cs_preds[i]
        cs = str(cs_preds[i])
        cs_proba = cs_preds_proba[i][key]

        pred_obj = {
            'id': match['id'],
            'hda': hda,
            'home_win_proba': home_win_proba,
            'draw_proba': draw_proba,
            'away_win_proba': away_win_proba,
            'bts': bts,
            'gg_proba': bts_proba[1],
            'ng_proba': bts_proba[0],
            'over25': over25,
            'over25_proba': over25_proba[1],
            'under25_proba': over25_proba[0],
            'cs_unsensored': cs,
            'cs_proba_unsensored': cs_proba,
        }

        predictions.append(pred_obj)

    data = {
        'version': '1.0',
        'type': 'regular',
        'competition_id': COMPETITION_ID,
        'date': str(target_date),
        'predictions': predictions}

    message = storePredictions(data, user_token)
    print(message)


def storePredictions(data, user_token):

    # Now that you have the user token, you can use it for other API requests.
    url = f"{API_BASE_URL}/admin/predictions/posting-from-python-app"

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
