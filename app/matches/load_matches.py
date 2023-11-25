from configs.settings import API_BASE_URL
from datetime import datetime
import json
import os
import pandas as pd
import requests

# Function to load data for all targets


def load_for_training(COMPETITION_ID, user_token, be_params, per_page=2000, train_ratio=.70, ignore_saved=False):

    from_date, to_date, history_limit_per_match = be_params

    from_date_str = from_date.strftime("%Y-%m-%d")
    to_date_str = to_date.strftime("%Y-%m-%d")

    dirname = os.path.dirname(__file__)
    filename = os.path.join(
        dirname, f"saved/{COMPETITION_ID}_matches.json")

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
        matches_url = f"{API_BASE_URL}/admin/competitions/view/{COMPETITION_ID}/matches?type=played&per_page={per_page}&from_date={from_date_str}&to_date={to_date_str}&with_stats=1&order_by=utc_date&order_direction=asc&history_limit_per_match={history_limit_per_match}"

        # Retrieve train and test match data
        all_matches = get(url=matches_url, user_token=user_token)

        # Save the fetched data to 'all_matches.json'
        with open(filename, 'w') as file:
            json.dump(all_matches, file)
    else:
        # Data found in json file, use it
        all_matches = loaded_results

    all_matches = add_features(all_matches)
    total_matches = len(all_matches)

    train_size = int(total_matches * train_ratio)

    # Split matches into train and test sets
    train_matches = all_matches[:train_size]
    test_matches = all_matches[train_size:]

    return train_matches, test_matches


def load_for_predictions(COMPETITION_ID, TARGET_DATE, user_token):

    # Now that you have the user token, you can use it for other API requests.
    url = f"{API_BASE_URL}/admin/competitions/view/{COMPETITION_ID}/matches?per_page=50&date={TARGET_DATE}&with_stats=1"

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
    all_matches = response.json()['results']['data']

    matches_data = []
    for match in all_matches:

        stats = match['stats']

        # We will filter if load_for_training, dont filter if load_for_predictions
        if filter == True and stats['target'] == False:
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
        }

        matches_data.append({**match, **current_match_data})

    return matches_data
