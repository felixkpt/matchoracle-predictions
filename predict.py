import argparse
from configs.settings import API_BASE_URL
from app.predictions.hda_predictions import hda_predictions
from app.predictions.bts_predictions import bts_predictions
from app.predictions.over_predictions import over_predictions
from app.predictions.cs_predictions import cs_predictions
from app.predictions.predictions_normalizer import predictions_normalizer as normalizer
from app.matches.load_matches import load_for_predictions
from configs.logger import Logger
from datetime import datetime, timedelta
import json
import requests
from configs.active_competitions.competition_data import get_trained_competitions, update_trained_competitions


def predict(user_token):
    print("\n............... START PREDICTIONS ..................\n")

    PREDICTION_TYPE = "regular_prediction_10_5_5"

    logger_model(user_token, PREDICTION_TYPE)

    parser = argparse.ArgumentParser(
        description='Run predictions for a specified competition.')
    parser.add_argument('--competition', type=int,
                        help='Competition ID for predictions')

    parser.add_argument('--ignore-timing', type=int,
                        help='Ignore last predicted check')

    args, extra_args = parser.parse_known_args()

    # If competition_id is provided, use it; otherwise, fetch from the backend API
    competition_ids = [
        args.competition] if args.competition is not None else get_trained_competitions(False, args.ignore_timing or False)

    for COMPETITION_ID in competition_ids:
        compe_data = {}
        compe_data['id'] = COMPETITION_ID
        compe_data['prediction_type'] = PREDICTION_TYPE

        Logger.info(f"Competition: {COMPETITION_ID}")
        Logger.info(f"Prediction type: {PREDICTION_TYPE}\n")
        do_competition_run_logging(user_token, PREDICTION_TYPE)

        plus_x_days = 16
        # Calculate today plus plus_x_days days
        today_plus_x_days = datetime.now() + timedelta(days=plus_x_days)
        # Convert today_plus_x_days to a string once
        today_plus_x_days_str = today_plus_x_days.strftime("%Y-%m-%d")
        # Calculate from_date and to_date
        from_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
        to_date = datetime.now() + timedelta(days=plus_x_days)

        data = {'results': {}}
        totals = 0

        # Iterate through a range of dates between from_date and to_date
        for i in range((to_date - from_date).days + 1):
            target_date = (from_date + timedelta(days=i))
            target_date_str = target_date.strftime("%Y-%m-%d")

            # Check if the target date is past today plus plus_x_days days
            if target_date_str > today_plus_x_days_str:
                Logger.info(
                    f"Aborting predictions for Competition {COMPETITION_ID} as target date {target_date} is past today plus {plus_x_days} days.")
                break

            Logger.info(f"Competition: {COMPETITION_ID}")
            Logger.info(f"Date: {target_date}\n")

            be_params = {
                'target_date': target_date_str,
                'prediction_type': PREDICTION_TYPE
            }

            matches = load_for_predictions(
                COMPETITION_ID, user_token, be_params)

            total_matches = len(matches)
            totals = totals + total_matches
            
            if total_matches == 0:
                print('No matches to make predictions!')
            else:
                print(f'Predicting {total_matches} matches...')

                hda_preds = hda_predictions(matches, compe_data)
                bts_preds = bts_predictions(matches, compe_data)
                over15_preds = over_predictions(
                    matches, compe_data, 'over15_target')
                over25_preds = over_predictions(
                    matches, compe_data, 'over25_target')
                over35_preds = over_predictions(
                    matches, compe_data, 'over35_target')
                cs_preds = cs_predictions(matches, compe_data)

                merge_and_store_predictions(user_token, compe_data, target_date, matches, hda_preds,
                                            bts_preds, over15_preds, over25_preds, over35_preds, cs_preds)


            print(f"______________\n")


        update_trained_competitions(user_token, compe_data)

        data['results']['saved_updated'] = totals
        do_logging(user_token, PREDICTION_TYPE, data=data)

        print(f"\n----- END COMPE PREDS -----")


    print(f"\n....... END PREDICTIONS, Happy coding! ........")


def merge_and_store_predictions(user_token, compe_data, target_date, matches, hda_preds, bts_preds, over15_preds, over25_preds, over35_preds, cs_preds):

    hda_preds, hda_preds_proba = hda_preds
    bts_preds, bts_preds_proba = bts_preds
    over15_preds, over15_preds_proba = over15_preds
    over25_preds, over25_preds_proba = over25_preds
    over35_preds, over35_preds_proba = over35_preds
    cs_preds, cs_preds_proba = cs_preds

    predictions = []
    for i, match in enumerate(matches):

        hda = str(hda_preds[i])
        hda_proba = hda_preds_proba[i]
        home_win_proba = hda_proba[0]
        draw_proba = hda_proba[1]
        away_win_proba = hda_proba[2]

        bts = str(bts_preds[i])
        bts_proba = bts_preds_proba[i]

        over15 = str(over15_preds[i])
        over15_proba = over15_preds_proba[i]

        over25 = str(over25_preds[i])
        over25_proba = over25_preds_proba[i]

        over35 = str(over35_preds[i])
        over35_proba = over35_preds_proba[i]

        cs = str(cs_preds[i])
        cs_proba = cs_preds_proba[i]

        prediction = {
            'id': match['id'],
            'hda': hda,
            'home_win_proba': home_win_proba,
            'draw_proba': draw_proba,
            'away_win_proba': away_win_proba,
            'bts': bts,
            'gg_proba': bts_proba[1],
            'ng_proba': bts_proba[0],
            'over15': over15,
            'over15_proba': over15_proba[1],
            'under15_proba': over15_proba[0],
            'over25': over25,
            'over25_proba': over25_proba[1],
            'under25_proba': over25_proba[0],
            'over35': over35,
            'over35_proba': over35_proba[1],
            'under35_proba': over35_proba[0],
            'cs': cs,
            'cs_proba': cs_proba,
        }

        prediction = normalizer(prediction, compe_data)

        predictions.append(prediction)

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


def do_competition_run_logging(user_token, PREDICTION_TYPE):
    url = f"{API_BASE_URL}/admin/predictions/from-python-app/predictions-job-logs"

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    # Assuming is_competition_run_logging is a boolean variable
    is_competition_run_logging = True

    json_data = json.dumps({
        "prediction_type": PREDICTION_TYPE,
        "is_competition_run_logging": is_competition_run_logging,
    })

    # Make a GET request with the headers
    response = requests.post(url, data=json_data, headers=headers)
    response.raise_for_status()


def do_logging(user_token, PREDICTION_TYPE, data=None):
    
    url = f"{API_BASE_URL}/admin/predictions/from-python-app/predictions-job-logs"

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    print(data['results'])
    json_data = json.dumps({
        "prediction_type": PREDICTION_TYPE,
        "results": data['results']
    })

    # Make a GET request with the headers
    response = requests.post(url, data=json_data, headers=headers)
    response.raise_for_status()


def logger_model(user_token, PREDICTION_TYPE=None):
    url = f"{API_BASE_URL}/admin/predictions/from-python-app/predictions-job-logs"

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    json_data = json.dumps({
        "prediction_type": PREDICTION_TYPE,
        "increment_job_run_counts": True,
    })

    # Make a GET request with the headers
    response = requests.post(url, data=json_data, headers=headers)
    response.raise_for_status()
