import requests
import json
import os
from app.configs.settings import API_BASE_URL, basepath
from app.helpers.functions import parse_json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

COMPETITION_API_URL = f"{API_BASE_URL}/dashboard/competitions?status=1&page=1&per_page=1000&order_direction=desc"


def get_competitions(user_token, games_counts_threshold=0):
    directory = os.path.abspath(os.path.join(
        basepath(), "configs/active_competitions/saved/"))
    os.makedirs(directory, exist_ok=True)

    # Save the features
    filename = os.path.abspath(f"{directory}/competition_data.json")

    try:
        # Check if the saved file exists and if it's less than an 1 day old
        if os.path.exists(filename):
            # Load the saved data
            with open(filename, 'r') as json_file:
                saved_data = json.load(json_file)

                if len(saved_data):
                    saved_time = datetime.strptime(
                        saved_data[0]["saved_at"], "%Y-%m-%d %H:%M:%S")
                    if datetime.now() - saved_time < timedelta(days=1):
                        # Return the competition IDs from the saved data
                        print('Retrieving saved compe IDS.')

                        filtered = []
                        for c in saved_data:
                            if c['games_counts'] >= games_counts_threshold:
                                filtered.append(c)
                        return filtered

        # Create a dictionary with the headers
        headers = {"Authorization": f"Bearer {user_token}"}

        # Make a GET request with the headers
        response = requests.get(COMPETITION_API_URL, headers=headers)
        response.raise_for_status()  # Raise an exception for bad responses (4xx and 5xx)
        data = response.json()

        # Get the current time
        current_time = datetime.now()

        competition_data = [{"id": competition["id"], "name": competition["country"]["name"]+' - '+competition["name"], "games_counts": competition["games_counts"], "saved_at": current_time.strftime("%Y-%m-%d %H:%M:%S")}
                            for competition in data["results"]['data']]

        # Save competition data to a JSON file
        with open(filename, 'w') as json_file:
            json.dump(competition_data, json_file)

        # Return the competition IDs after saving to the file
        print('Retrieving compe IDS from backend.')

        filtered = []
        for c in competition_data:
            if c['games_counts'] >= games_counts_threshold:
                filtered.append(c)

        return filtered

    except requests.RequestException as e:
        print(f"Error fetching competition data: {e}")
        return []


def update_trained_competitions(user_token, compe_data, train_matches_counts, start_time):
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

    print('Updating trained:', compe_data)
    id = compe_data['id']
    games_counts = compe_data['games_counts']

    current_datetime = datetime.today()

    now = None if train_matches_counts < 10 else current_datetime.strftime(
        '%Y-%m-%d %H:%M:%S')

    if now is None:
        compe_data['trained_to'] = None

    # Check if the competition ID already exists in the trained competition data
    if id in trained_compe_data:
        # Update only the competition training timestamp
        trained_compe_data[id]["competition_trained_at"] = now
        trained_compe_data[id]["last_predicted_at"] = None
        do_update_trained_competition(user_token, compe_data, train_matches_counts, start_time)
    else:
        # If the competition ID is not found, add a new entry with current timestamps
        trained_compe_data[id] = {
            "games_counts": games_counts,
            "competition_trained_at": now,
            "last_predicted_at": None
        }
        do_update_trained_competition(user_token, compe_data, train_matches_counts, start_time)

    with open(filename, 'w') as file:
        json.dump(trained_compe_data, file, indent=4)


def update_last_predicted_at(user_token,compe_data):
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

    id = int(compe_data['id'])
    current_datetime = datetime.today()
    now = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Update the competition data with the current timestamp for last prediction
    trained_compe_data[id] = {
        "competition_trained_at": trained_compe_data[id]['competition_trained_at'],
        "last_predicted_at": now
    }

    with open(filename, 'w') as file:
        json.dump(trained_compe_data, file, indent=4)


def get_trained_competitions(last_action_date=None, is_train=False):
    """
    Get trained competitions based on the last predicted date.

    Args:
        last_action_date (str, optional): Last training date in the format "%Y-%m-%d %H:%M:%S". Defaults to None but last 3 days will used internally.

    Returns:
        list: List of competition IDs.
    """

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

    last_action_date = last_action_date if last_action_date is not None else (
        datetime.today() + relativedelta(days=-3)).strftime("%Y-%m-%d %H:%M:%S")

    # Filter out competitions based on timing
    if is_train == True:

        filtered_competitions = {
            competition_id: competition_info
            for competition_id, competition_info in trained_compe_data.items()
            if (
                competition_info["competition_trained_at"] is not None and
                datetime.strptime(
                    competition_info["competition_trained_at"], "%Y-%m-%d %H:%M:%S") >= datetime.strptime(last_action_date, "%Y-%m-%d %H:%M:%S")
            )
        }
    else:
        filtered_competitions = {
            competition_id: competition_info
            for competition_id, competition_info in trained_compe_data.items()
            if (
                # Check if competition_trained_at is available
                competition_info.get("competition_trained_at") is not None and
                (
                    competition_info["last_predicted_at"] is None or
                    datetime.strptime(
                        competition_info["last_predicted_at"], "%Y-%m-%d %H:%M:%S") <= datetime.strptime(last_action_date, "%Y-%m-%d %H:%M:%S")
                )
            )
        }
    
    sort_by = "competition_trained_at" if is_train else "last_predicted_at"
    # Sort the filtered competitions by last_predicted_at timestamp in descending order (newest first)
    # and put None values at the top (ascending order)
    sorted_filtered_competitions = dict(sorted(
        filtered_competitions.items(),
        key=lambda x: (
            filtered_competitions[x[0]].get(sort_by, "") is None,
            filtered_competitions[x[0]].get(sort_by, "")
        ),
        reverse=True
    ))

    return sorted_filtered_competitions


def do_update_trained_competition(user_token, compe_data, train_matches_counts, start_time):
    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    # End the timer
    end_time = datetime.now()
    duration = end_time - start_time
    duration = duration.total_seconds()

    json_data = json.dumps({
        "competition_id": compe_data['id'],
        "trained_to": compe_data['trained_to'],
        "prediction_type": compe_data['prediction_type'],
        "results": {"created_counts": train_matches_counts, "updated_counts": 0, "failed_counts": 0},
        "seconds_taken": duration,
        "status": 200,
    })

    url = f"{API_BASE_URL}/dashboard/predictions/from-python-app/update-competition-last-training"

    response = requests.post(url, data=json_data, headers=headers)
    print(response.text)
    response.raise_for_status()

def do_update_predicted_competition(user_token, compe_data, start_time):
    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    # End the timer
    end_time = datetime.now()
    duration = end_time - start_time
    duration = duration.total_seconds()

    json_data = json.dumps({
        "competition_id": compe_data['id'],
        "prediction_type": compe_data['prediction_type'],
        "results": {"created_counts": len(compe_data['predictions']), "updated_counts": 0, "failed_counts": 0},
        "seconds_taken": duration,
        "status": 200,
    })

    url = f"{API_BASE_URL}/dashboard/predictions/from-python-app/update-competition-last-prediction"

    response = requests.post(url, data=json_data, headers=headers)
    print(response.text)
    response.raise_for_status()

def update_job_status(user_token, job_id, status="completed"):
    """
    Updates the job status for the given job_id.
    :param user_token: Token for authorization.
    :param job_id: ID of the job to update.
    :param status: The new status to set (default: "completed").
    :return: Response from the API.
    """
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    payload = json.dumps({
        "status": status
    })

    url = f"{API_BASE_URL}/dashboard/jobs/{job_id}/update-status"
    
    response = requests.patch(url, data=payload, headers=headers)
    
    if response.status_code == 200:
        print(f"Job {job_id} status updated to '{status}'.")
    else:
        print(f"Failed to update status for job {job_id}. Response: {response.text}")

    response.raise_for_status()

    return response
