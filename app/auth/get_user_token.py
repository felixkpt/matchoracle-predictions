import requests
from app.configs.settings import API_BASE_URL


def get_user_token(email, password):
    # The API endpoint for user login
    login_url = f"{API_BASE_URL}/auth/login"

    # Make a request to obtain the user token
    data = {
        'email': email,
        'password': password,
    }

    response = requests.post(login_url, data=data)

    if response.status_code == 200:
        response_json = response.json()
        user_token = response_json.get('results', {}).get('token')
        return user_token
    else:
        return None
