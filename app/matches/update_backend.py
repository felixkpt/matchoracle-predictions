from app.configs.settings import API_BASE_URL
import json
import requests


def update_backend(user_token, COMPETITION_ID, target, main_object):

    # Now that you have the user token, you can use it for other API requests.
    url = f"{API_BASE_URL}/predictions/from-python-app/store-competition-score-target-outcome"

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
        'Content-Type': 'application/json',
    }

    data = {
        **{"competition_id": COMPETITION_ID, 'score_target_outcome_id': target, }, ** main_object
    }

    json_data = json.dumps(data)
    # Make a GET request with the headers
    response = requests.post(url, data=json_data, headers=headers)
    response.raise_for_status()

    # Parse the JSON response
    message = response.json()['message']

    return message
