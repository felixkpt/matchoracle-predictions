from configs.settings import API_BASE_URL
from datetime import datetime
import json
import os
import pandas as pd
import requests

# Function to load data for all targets


def load_for_training(compe_data, user_token, be_params, per_page=1500, train_ratio=.70, ignore_saved=False):
    COMPETITION_ID = compe_data['id']
    PREDICTION_TYPE = compe_data['prediction_type']

    to_date_str = be_params['to_date'].strftime("%Y-%m-%d")
    history_limit_per_match = be_params['history_limit_per_match']
    current_ground_limit_per_match = be_params['current_ground_limit_per_match']
    h2h_limit_per_match = be_params['h2h_limit_per_match']

    # Create the directory if it doesn't exist
    directory = os.path.abspath(
        f"app/matches/saved/{PREDICTION_TYPE}/")
    os.makedirs(directory, exist_ok=True)

    # Save the features
    filename = os.path.abspath(f"{directory}/{COMPETITION_ID}_matches.json")

    loaded_results = None
    if not ignore_saved:
        print(f"Getting saved matches...")
        try:
            # Try to load data from json file
            with open(filename, 'r') as file:
                loaded_results = json.load(file)
        except:
            FileNotFoundError

    if ignore_saved or loaded_results is None:
        print(f"Getting matches with stats from BE...")

        # Construct the URL for train and test data for the current target
        matches_url = f"{API_BASE_URL}/admin/competitions/view/{COMPETITION_ID}/matches?type=past&per_page={per_page}&to_date={to_date_str}&is_predictor=1&task=train&order_by=utc_date&order_direction=desc&history_limit_per_match={history_limit_per_match}&current_ground_limit_per_match={current_ground_limit_per_match}&h2h_limit_per_match={h2h_limit_per_match}"

        # Retrieve train and test match data
        all_matches = get(url=matches_url, user_token=user_token)
        
        # Save the fetched data to 'all_matches.json'
        with open(filename, 'w') as file:
            json.dump(all_matches, file)
    else:
        # Data found in json file, use it
        all_matches = loaded_results

    all_matches = add_features(all_matches, key='utc_date')
    total_matches = len(all_matches)
    
    if total_matches < 50: return [], []

    train_size = int(total_matches * train_ratio)

    # Split matches into train and test sets
    train_matches = all_matches[:train_size]
    test_matches = all_matches[train_size:]

    return train_matches, test_matches


def load_for_predictions(COMPETITION_ID, user_token, be_params):

    target_date = be_params['target_date']
    prediction_type = be_params['prediction_type']

    # Now that you have the user token, you can use it for other API requests.
    url = f"{API_BASE_URL}/admin/competitions/view/{COMPETITION_ID}/matches?per_page=50&date={target_date}&is_predictor=1&task=predict&order_by=utc_date&order_direction=desc&prediction_type={prediction_type}"

    matches_data = get(url=url, user_token=user_token, filter=False)

    all_matches = add_features(matches_data)

    return all_matches


def get(url, user_token, filter=True):

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
    }

    # Make a GET request with the headers
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error if the request was not successful

    # Parse the JSON response
    all_matches = response.json()

    matches_data = []
    for match in all_matches:
        stats = {}
        if 'stats' in match:
            stats = match['stats']
            del match['stats']
            del match['competition']
            del match['home_team']
            del match['away_team']
            del match['score']

            if stats == None:
                continue

            # We will filter if load_for_training, dont filter if load_for_predictions
            if filter == True and not stats['has_results']:
                continue
        elif match['cs_target'] == '-':
            continue

        matches_data.append({**match, **stats})

    return matches_data


def add_features(all_matches, key='utc_date'):
    matches_data = []
    for match in all_matches:

        # Convert the string to a datetime object
        current_match_data = {}
        if key in match:
            utc_date = datetime.strptime(
                match[key], "%Y-%m-%d %H:%M:%S")
            date = pd.to_datetime(utc_date)

            hour = utc_date.hour
            day_of_week = utc_date.weekday()

            current_match_data = {
                "date": date,
                "hour": hour,
                "day_of_week": day_of_week,
            }

        matches_data.append({**match, **current_match_data})

    return matches_data
