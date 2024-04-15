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
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import requests
import argparse
from configs.active_competitions.competitions_data import get_trained_competitions
from app.predictions_normalizers.predictions_normalizer import predictions_normalizer
from configs.active_competitions.competitions_data import update_last_predicted_at


def predict(user_token):
    """
    Function to predict match outcomes based on various configurations.

    Args:
        user_token (str): User token for authentication.

    Returns:
        None
    """

    print("\n............... START PREDICTIONS ..................\n")

    # Set the prediction type
    PREDICTION_TYPE = f"regular_prediction_12_6_4_1200"

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predictions with different configurations.')
    parser.add_argument('--competition', type=int, help='Competition ID')
    parser.add_argument('--target', choices=['hda', 'ft_hda', 'ht_hda', 'bts', 'over15', 'over25', 'over35', 'cs'],
                        help='Target for predictions')
    parser.add_argument('--last-predict-date', help='Last predict date')

    parser.add_argument('--from-date', type=str, help='From date')
    parser.add_argument('--to-date', type=str, help='To date')
    parser.add_argument('--target-match', type=int, help='Match ID')

    args, extra_args = parser.parse_known_args()
    target = args.target
    last_action_date = args.last_predict_date

    from_date = args.from_date
    to_date = args.to_date
    target_match = args.target_match

    print(f"Main Prediction Target: {target if target else 'all'}")

    # Set last_action_date dynamically
    last_action_date = last_action_date if last_action_date is not None else (datetime.now() - timedelta(hours=24 * 3)
                                                                                            ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Performing preds on unpredicted competitions or those predicted on/before: {last_action_date}")

    # Calculate from_date and to_date
    from_date = datetime.strptime(
        from_date, '%Y-%m-%d') if from_date else datetime.today() + relativedelta(days=-30 * 6)
    to_date = datetime.strptime(
        to_date, '%Y-%m-%d') if to_date else datetime.today() + relativedelta(days=7)

    print(f"From & to date: {from_date}, {to_date}\n")

    # If competition_id is provided, use it; otherwise, fetch from the backend API
    competition_ids = [
        args.competition] if args.competition is not None else get_trained_competitions(last_action_date)

    # Loop over competition IDs
    for i, COMPETITION_ID in enumerate(competition_ids):
        compe_data = {}
        compe_data['id'] = COMPETITION_ID
        compe_data['prediction_type'] = PREDICTION_TYPE

        Logger.info(
            f"{i+1}/{len(competition_ids)}. Competition: #{COMPETITION_ID}")
        Logger.info(f"Prediction type: {PREDICTION_TYPE}")

        dates = get_dates_with_games(user_token, COMPETITION_ID, from_date.strftime(
            "%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))

        print(f'Dates with games in selected range: {len(dates)}\n')

        # Loop through each day from from_date to to_date
        for target_date in dates:
            Logger.info(f"Competition: {COMPETITION_ID}")
            Logger.info(f"Date: {target_date}\n")

            matches = load_for_predictions(
                COMPETITION_ID, target_date, user_token)

            total_matches = len(matches)

            if total_matches == 0:
                print('No matches to make predictions!')
            else:
                print(f'Predicting {total_matches} matches...')

                # Get predictions for different outcomes
                ft_hda_preds = ft_hda_predictions(matches, compe_data)
                ht_hda_preds = ht_hda_predictions(matches, compe_data)
                bts_preds = bts_predictions(matches, compe_data)
                over15_preds = over15_predictions(matches, compe_data)
                over25_preds = over25_predictions(matches, compe_data)
                over35_preds = over35_predictions(matches, compe_data)
                cs_preds = cs_predictions(matches, compe_data)

                # Check if any of the required predictions is null
                if ft_hda_preds[0] is not None or over15_preds[0] is not None or over25_preds[0] is not None or over35_preds[0] is not None or bts_preds[0] is not None is not None or cs_preds[0] is not None:
                    # Merge and store predictions
                    merge_and_store_predictions(user_token, compe_data, target_date, matches, target_match, ft_hda_preds, ht_hda_preds,
                                                bts_preds, over15_preds, over25_preds, over35_preds, cs_preds)
                    # Update last predicted competitions
                    update_last_predicted_at(compe_data)
                else:
                    print('One of the required preds is null.')

            print(f"______________\n")

        print(f"--- End preds for compe #{COMPETITION_ID} ---\n")

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
        'version': '1.0',
        'type': compe_data['prediction_type'],
        'competition_id': compe_data['id'],
        'date': str(target_date),
        'predictions': predictions
    }

    if len(data['predictions']) > 0:
        message = storePredictions(data, user_token)
        print(message)


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
    url = f"{API_BASE_URL}/admin/predictions/from-python-app/store-predictions"

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
    url = f"{API_BASE_URL}/admin/competitions/view/{COMPETITION_ID}/get-dates-with-games"

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
