import requests
import json
import os
from configs.settings import API_BASE_URL, basepath
from app.helpers.functions import parse_json
from datetime import datetime, timedelta

COMPETITION_API_URL = f"{API_BASE_URL}/admin/competitions?active_only=1&page=1&per_page=1000&order_direction=desc"


def get_competition_ids(user_token):
    # Save the features
    filename = os.path.abspath(
        os.path.join(basepath(), "configs/active_competitions/saved/competition_data.json"))

    try:
        # Read the JSON file
        with open(filename, 'r') as json_file:
            competition_data = json.load(json_file)

        # Extract competition IDs
        competition_ids = [competition["id"]
                           for competition in competition_data]

        return competition_ids
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            # Create a dictionary with the headers
            headers = {
                "Authorization": f"Bearer {user_token}",
            }

            # Make a GET request with the headers
            response = requests.get(COMPETITION_API_URL, headers=headers)
            response.raise_for_status()  # Raise an exception for bad responses (4xx and 5xx)
            data = response.json()

            competition_data = [{"id": competition["id"], "name": competition["name"]}
                                for competition in data["results"]['data']]

            # Save competition data to a JSON file
            with open(filename, 'w') as json_file:
                json.dump(competition_data, json_file)

            # Return the competition IDs after saving to the file
            return [competition["id"] for competition in competition_data]

        except requests.RequestException as e:
            print(f"Error fetching competition data: {e}")
            return []


def trained_competitions(user_token, compe_data):
    directory = os.path.abspath(
        os.path.join(basepath(), "configs/active_competitions/saved/"))
    os.makedirs(directory, exist_ok=True)

    filename = os.path.abspath(
        os.path.join(directory, "trained_competitions.json"))

    try:
        with open(filename, 'r') as file:
            trained_compe_data = parse_json(json.load(file))
    except FileNotFoundError:
        trained_compe_data = {}

    id = compe_data['id']

    current_datetime = datetime.today()
    now = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Check if the competition ID already exists in the trained competition data
    if id in trained_compe_data:
        # Update only the competition training timestamp
        trained_compe_data[id]["competition_trained_at"] = now
        do_update_trained_competition(user_token, compe_data)
    else:
        # If the competition ID is not found, add a new entry with current timestamps
        trained_compe_data[id] = {
            "competition_trained_at": now,
            "last_predicted_at": None
        }
        do_update_trained_competition(user_token, compe_data)

    with open(filename, 'w') as file:
        json.dump(trained_compe_data, file, indent=4)


def update_trained_competitions(compe_data):
    directory = os.path.abspath(
        os.path.join(basepath(), "configs/active_competitions/saved/"))
    os.makedirs(directory, exist_ok=True)

    filename = os.path.abspath(
        os.path.join(directory, "trained_competitions.json"))

    try:
        with open(filename, 'r') as file:
            trained_compe_data = parse_json(json.load(file))
    except FileNotFoundError:
        trained_compe_data = {}

    id = compe_data['id']
    current_datetime = datetime.today()
    now = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Update the competition data with the current timestamp for last prediction
    trained_compe_data[id] = {
        "competition_trained_at": trained_compe_data[id]['competition_trained_at'],
        "last_predicted_at": now
    }

    with open(filename, 'w') as file:
        json.dump(trained_compe_data, file, indent=4)


def get_trained_competitions(ignore_filters=False, ignore_timimg=False):
    directory = os.path.abspath(
        os.path.join(basepath(), "configs/active_competitions/saved/"))
    os.makedirs(directory, exist_ok=True)

    filename = os.path.abspath(
        os.path.join(directory, "trained_competitions.json"))

    try:
        with open(filename, 'r') as file:
            trained_compe_data = parse_json(json.load(file))
    except FileNotFoundError:
        trained_compe_data = {}

    if ignore_filters:
        # If ignore_filters is True, return all competition IDs without applying the time-based filter
        return trained_compe_data.keys()

    # Get the current time
    current_time = datetime.now()

    # Filter out competitions with last_predicted_at less than 3 hours ago
    filtered_competitions = {
        competition_id: competition_info
        for competition_id, competition_info in trained_compe_data.items()
        if (
            competition_info.get("last_predicted_at") is None
            or current_time - datetime.strptime(competition_info["last_predicted_at"], "%Y-%m-%d %H:%M:%S") >= timedelta(hours=0 if ignore_timimg else 24 * 3)
        )
    }

    # Sort the filtered competitions by last_predicted_at timestamp in descending order (newest first)
    # and put None values at the top (ascending order)
    sorted_filtered_competitions = dict(sorted(
        filtered_competitions.items(),
        key=lambda x: (
            filtered_competitions[x[0]].get("last_predicted_at", "") is None,
            filtered_competitions[x[0]].get("last_predicted_at", "")
        ),
        reverse=True
    ))

    return sorted_filtered_competitions.keys()


def do_update_trained_competition(user_token, compe_data):
    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    json_data = json.dumps({
        "competition_id": compe_data['id'],
        "trained_to": compe_data['trained_to']
    })

    url = f"{API_BASE_URL}/admin/predictions/from-python-app/update-competition-last-training"

    response = requests.post(url, data=json_data, headers=headers)
    print(response.text)
    response.raise_for_status()
