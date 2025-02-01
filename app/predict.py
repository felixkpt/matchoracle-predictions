from app.configs.settings import API_BASE_URL
from app.predictions.ft_hda_predictions import ft_hda_predictions
from app.predictions.ht_hda_predictions import ht_hda_predictions
from app.predictions.bts_predictions import bts_predictions
from app.predictions.over15_predictions import over15_predictions
from app.predictions.over25_predictions import over25_predictions
from app.predictions.over35_predictions import over35_predictions
from app.predictions.cs_predictions import cs_predictions
from app.matches.load_matches import load_for_predictions
from app.configs.logger import Logger
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import requests
from app.configs.active_competitions.competitions_data import get_trained_competitions
from app.predictions_normalizers.predictions_normalizer import predictions_normalizer
from app.configs.active_competitions.competitions_data import update_last_predicted_at
from dateutil import parser
from app.configs.active_competitions.competitions_data import update_job_status
from app.configs.active_competitions.competitions_data import do_update_predicted_competition


async def predict(user_token, prediction_type, request_data):
    """
    Function to predict match outcomes based on various configurations.

    Args:
        user_token (str): User token for authentication.

    Returns:
        None
    """

    print("\n............... START PREDICTIONS ..................\n")

    # Set the prediction type
    PREDICTION_TYPE = prediction_type
    VERSION = '1.0'

    # Extract parameters from request_data
    competition_id = request_data.get('competition')
    target = request_data.get('target')
    last_action_date = request_data.get('last_predict_date')
    from_date = request_data.get('from_date')
    to_date = request_data.get('to_date')
    target_match = request_data.get('target_match')

    print(f"Main Prediction Target: {target if target else 'all'}")

    # Set last_action_date dynamically
    last_action_date = last_action_date if last_action_date is not None else (datetime.now() - timedelta(hours=24 * 3)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Performing preds on unpredicted competitions or those predicted on/before: {last_action_date}")


    # Calculate from_date and to_date
    from_date = parser.parse(from_date) if from_date else datetime.today() + relativedelta(days=-30 * 0)
    to_date = parser.parse(to_date) if to_date else datetime.today() + relativedelta(days=7)
    
    print(f"From & to date: {from_date}, {to_date}\n")

    # If competition_id is provided, use it; otherwise, fetch from the backend API
    competitions = {f"{competition_id}": {'id': competition_id, 'last_predicted_at': 'N/A'}} if competition_id else get_trained_competitions(last_action_date)

    # Loop over competition IDs
    for i, COMPETITION_ID in enumerate(competitions):

        compe_data = {}
        compe_data['id'] = COMPETITION_ID
        compe_data['prediction_type'] = PREDICTION_TYPE
        compe_data['version'] = VERSION
        compe_data['predictions'] = []

        Logger.info(
            f"{i+1}/{len(competitions)}. Competition: #{COMPETITION_ID}, (last pred. {competitions[COMPETITION_ID]['last_predicted_at']} )")
        Logger.info(f"Prediction type: {PREDICTION_TYPE}")

        dates = get_dates_with_games(user_token, COMPETITION_ID, from_date.strftime(
            "%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))

        print(f'Dates with predictable games in selected range: {len(dates)}\n')
        # Start the timer
        start_time = datetime.now()

        # Loop through each day from from_date to to_date
        for target_date in dates:
            Logger.info(f"Competition: {COMPETITION_ID}")
            Logger.info(f"Date: {target_date}\n")

            matches = load_for_predictions(user_token, compe_data, target_date)

            total_matches = len(matches)

            if total_matches == 0:
                print('No matches to make predictions!')
            else:
                print(f'Predicting {total_matches} matches...')

                # Get predictions for different outcomes
                ft_hda_preds = ht_hda_preds = bts_preds = over15_preds= over25_preds = over35_preds = cs_preds = [None, None]
                if target is None or target == 'hda' or target == 'ft-hda':
                    ft_hda_preds = ft_hda_predictions(matches, compe_data)
                if target is None or target == 'ht-hda':
                    ht_hda_preds = ht_hda_predictions(matches, compe_data)
                if target is None or target == 'bts':
                    bts_preds = bts_predictions(matches, compe_data)
                if target is None or target == 'over15':
                    over15_preds = over15_predictions(matches, compe_data)
                if target is None or target == 'over25':
                    over25_preds = over25_predictions(matches, compe_data)
                if target is None or target == 'over35':
                    over35_preds = over35_predictions(matches, compe_data)
                if target is None or target == 'cs':
                    cs_preds = cs_predictions(matches, compe_data)

                # Check if any of the required predictions is null
                if ft_hda_preds[0] is not None or over15_preds[0] is not None or over25_preds[0] is not None or over35_preds[0] is not None or bts_preds[0] is not None is not None or cs_preds[0] is not None:
                    # Merge and store predictions
                    predictions = merge_and_store_predictions(user_token, compe_data, target_date, matches, target_match, ft_hda_preds, ht_hda_preds,
                                                bts_preds, over15_preds, over25_preds, over35_preds, cs_preds)
                    compe_data['predictions'] = predictions
                    # Update last predicted competitions
                    update_last_predicted_at(user_token,compe_data)
                else:
                    print('One of the required preds is null.')

            print(f"______________\n")

        do_update_predicted_competition(user_token, compe_data, start_time)
        
        print(f"--- End preds for compe #{COMPETITION_ID} ---\n")


    job_id = request_data.get('job_id')
    print('job_id:', job_id)
    if job_id:
        update_job_status(user_token, job_id, status="completed")

    print(f"\n....... END PREDICTIONS, Happy coding! ........")


def merge_and_store_predictions(user_token, compe_data, target_date, matches, target_match, ft_hda_preds, ht_hda_preds,
                                bts_preds, over15_preds, over25_preds, over35_preds, cs_preds):
    """
    Merge predictions and store them.

    Args:
        user_token (str): User token for authentication.
        compe_data (dict): Competition data.
        target_date (str): Target date for predictions.
        matches (list): List of matches.
        ft_hda_preds (tuple): Full-time HDA predictions.
        ht_hda_preds (tuple): Half-time HDA predictions.
        bts_preds (tuple): Both teams to score predictions.
        over15_preds (tuple): Over 1.5 goals predictions.
        over25_preds (tuple): Over 2.5 goals predictions.
        over35_preds (tuple): Over 3.5 goals predictions.
        cs_preds (tuple): Correct score predictions.

    Returns:
        None
    """

    ft_hda_preds, ft_hda_preds_proba = ft_hda_preds
    ht_hda_preds, ht_hda_preds_proba = ht_hda_preds
    bts_preds, bts_preds_proba = bts_preds
    over15_preds, over15_preds_proba = over15_preds
    over25_preds, over25_preds_proba = over25_preds
    over35_preds, over35_preds_proba = over35_preds
    cs_preds, cs_preds_proba = cs_preds

    predictions = []
    print(len(matches))

    for i, match in enumerate(matches):
        print('Match ID:', match['id'])
        if target_match and match['id'] != target_match:
            continue

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
        if ht_hda_preds_proba and len(ht_hda_preds_proba):
            try:
                ht_hda = str(ht_hda_preds[i])
                hda_proba = ht_hda_preds_proba[i]
                ht_home_win_proba = hda_proba[0]
                ht_draw_proba = hda_proba[1]
                ht_away_win_proba = hda_proba[2]
            except TypeError:
                pass

        bts = str(bts_preds[i])
        bts_proba = bts_preds_proba[i]

        over15 = str(over15_preds[i])
        over15_proba = over15_preds_proba[i]

        over25 = str(over25_preds[i])
        over25_proba = over25_preds_proba[i]

        over35 = str(over35_preds[i])
        over35_proba = over35_preds_proba[i]

        cs = str(cs_preds[i])
        cs_proba = max(cs_preds_proba[i])

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
            'over_under15_pick': over15,
            'over15_proba': over15_proba[1],
            'under15_proba': over15_proba[0],
            'over_under25_pick': over25,
            'over25_proba': over25_proba[1],
            'under25_proba': over25_proba[0],
            'over_under35_pick': over35,
            'over35_proba': over35_proba[1],
            'under35_proba': over35_proba[0],
            'cs': cs,
            'cs_proba': cs_proba,
        }

        # Normalize predictions
        pred_obj = predictions_normalizer(pred_obj, compe_data)
        predictions.append(pred_obj)

    data = {
        'prediction_type': compe_data['prediction_type'],
        'version': compe_data['version'],
        'competition_id': compe_data['id'],
        'date': str(target_date),
        'predictions': predictions
    }

    if len(data['predictions']) > 0:
        message = storePredictions(data, user_token)
        print(message)
    
    return predictions


def storePredictions(data, user_token):
    """
    Store predictions to the backend API.

    Args:
        data (dict): Prediction data.
        user_token (str): User token for authentication.

    Returns:
        str: Message indicating the success of the operation.
    """

    # Set the API endpoint URL
    url = f"{API_BASE_URL}/predictions/from-python-app/store-predictions"

    # Set headers for the request
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    json_data = json.dumps(data)
    # Make a POST request to store predictions
    response = requests.post(url, data=json_data, headers=headers)
    response.raise_for_status()

    # Parse the JSON response
    message = response.json()['message']

    return message


def get_dates_with_games(user_token, COMPETITION_ID, from_date, to_date):
    """
    Fetch dates with games for a specific competition within a date range.

    Args:
        user_token (str): User token for authentication.
        COMPETITION_ID (int): Competition ID.
        from_date (str): Start date.
        to_date (str): End date.

    Returns:
        list: List of dates with games.
    """

    # Set the API endpoint URL
    url = f"{API_BASE_URL}/competitions/view/{COMPETITION_ID}/get-dates-with-unpredicted-games"

    # Set headers for the request
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    # Prepare data for the request
    data = {
        'from_date': from_date,
        'to_date': to_date,
    }

    json_data = json.dumps(data)
    # Make a GET request to fetch dates with games
    response = requests.get(url, data=json_data, headers=headers)
    response.raise_for_status()

    # Parse the JSON response
    dates = response.json()['results']

    return dates
