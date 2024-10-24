from app.configs.settings import API_BASE_URL, basepath
from datetime import datetime
import json
import os
import pandas as pd
import requests

# Function to load data for all targets


def load_for_training(user_token, compe_data, target, be_params, per_page=2000, train_ratio=.70, ignore_saved_matches=False):
    COMPETITION_ID = compe_data.get('id')
    PREDICTION_TYPE = compe_data.get('prediction_type')

    history_limit_per_match = be_params.get('history_limit_per_match')
    current_ground_limit_per_match = be_params.get(
        'current_ground_limit_per_match')
    h2h_limit_per_match = be_params.get('h2h_limit_per_match')
    to_date = be_params.get('to_date')

    to_date_str = to_date.strftime("%Y-%m-%d")

    directory = os.path.abspath(
        os.path.join(basepath(), "matches/saved/"))
    os.makedirs(directory, exist_ok=True)

    filename = os.path.abspath(os.path.join(
        directory, f"{COMPETITION_ID}_matches.json"))

    loaded_results = None
    if not ignore_saved_matches:
        print(f"Getting saved matches...")
        try:
            # Try to load data from json file
            with open(filename, 'r') as file:
                loaded_results = json.load(file)
        except:
            FileNotFoundError

    if ignore_saved_matches or loaded_results is None:
        print(f"Getting matches with stats from BE...")

        # Construct the URL for train and test data for the current target
        matches_url = f"{API_BASE_URL}/dashboard/competitions/view/{COMPETITION_ID}/matches?type=played&per_page={per_page}&to_date={to_date_str}&is_predictor=1&order_by=utc_date&order_direction=asc&history_limit_per_match={history_limit_per_match}&current_ground_limit_per_match={current_ground_limit_per_match}&h2h_limit_per_match={h2h_limit_per_match}&prediction_type={PREDICTION_TYPE}&task=train"

        print(matches_url)
        # Retrieve train and test match data
        all_matches = get(url=matches_url, user_token=user_token)

        # Save the fetched data to 'all_matches.json'
        with open(filename, 'w') as file:
            json.dump(all_matches, file)
    else:
        # Data found in json file, use it
        all_matches = loaded_results

    if target == 'ht_hda_target':
        all_matches = [m for m in all_matches if m['ht_hda_target'] >= 0]

    all_matches = add_features(all_matches)
    total_matches = len(all_matches)

    if total_matches < 50:
        all_matches = []

    train_size = int(total_matches * train_ratio)

    # Split matches into train and test sets
    train_matches = all_matches[:train_size]
    test_matches = all_matches[train_size:]

    return train_matches, test_matches


def load_for_predictions(user_token, compe_data, TARGET_DATE):
    COMPETITION_ID = compe_data.get('id')
    PREDICTION_TYPE = compe_data.get('prediction_type')
    VERSION = compe_data.get('version')

    # Now that you have the user token, you can use it for other API requests.
    url = f"{API_BASE_URL}/dashboard/competitions/view/{COMPETITION_ID}/matches?per_page=50&date={TARGET_DATE}&prediction_type={PREDICTION_TYPE}&version={VERSION}&is_predictor=1&task=predict"
    
    matches_data = get(url=url, user_token=user_token, filter=False)

    all_matches = add_features(matches_data)

    return all_matches


def get(url, user_token, filter=True):

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Make a GET request with the headers
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error if the request was not successful

    # Parse the JSON response
    all_matches = response.json()

    matches_data = []
    for match in all_matches:
        try:
            if isinstance(match, dict):
                stats = match['stats']
            else:
                raise TypeError("Expected 'match' to be a dictionary.")
        except TypeError as e:
            print(f"Error: {e}")
            print("Problematic match:", match)
            continue
        
        if not stats:
            continue

        # We will filter if load_for_training, dont filter if load_for_predictions
        if filter == True and not stats['has_results']:
            continue

        matches_data.append({**match, **stats})

    return matches_data


def add_features(all_matches):
    matches_data = []
    for match in all_matches:

        # Convert the string to a datetime object
        utc_date = datetime.strptime(
            match['utc_date'], "%Y-%m-%d %H:%M:%S")
        date = pd.to_datetime(utc_date)

        hour = utc_date.hour
        day_of_week = utc_date.weekday()

        current_match_data = {
            "date": date,
            "hour": hour,
            "day_of_week": day_of_week,
            "season_id": match['season_id'] or 0
        }

        matches_data.append({**match, **current_match_data})

    return matches_data
