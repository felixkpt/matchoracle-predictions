import requests
from datetime import datetime
from composer import Composer
import match_utils
from team_stats import calculate_team_stats
import pandas as pd


def get_and_do_stats(url, API_BASE_URL, user_token, game_date, target = 'h'):
    # Parse the JSON response

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
    }

    # Make a GET request with the headers
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error if the request was not successful

    # Parse the JSON response
    train_matches = response.json()['results']['data']

    matches_data = []
    for match in train_matches:

        # Now that you have the user token, you can use it for other API requests.
        url = f"{API_BASE_URL}/admin/matches/view/{match['id']}"

        # Make a GET request with the headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error if the request was not successful

        # Parse the JSON response
        response_json = response.json()

        # Extract the data from the response
        matchData = response_json['results']['data']

        # Extract specific data you need
        # Convert the string to a datetime object
        utc_date = datetime.strptime(
            matchData['utc_date'], "%Y-%m-%d %H:%M:%S")
        date = pd.to_datetime(utc_date)

        hour = utc_date.hour
        day_of_week = utc_date.weekday()

        competition = matchData['competition']
        season_id = matchData['season_id']
        country_id = matchData['competition']['country_id']

        home_team = matchData['home_team']
        home_team_id = home_team['id']

        away_team = matchData['away_team']
        away_team_id = away_team['id']

        winner = Composer.winning_side(game=matchData, integer=True)
        goals = Composer.goals(game=matchData, integer=True)
        gg = Composer.gg(game=matchData, integer=True)

        # we only want matches with regular results
        if (winner != 1 and winner != 0 and winner == 2): continue

        # print("Competition:", competition['name'])
        # print("Season ID:", season_id)
        # print("Country ID:", country_id)
        # print("Home Team:", home_team['name'])
        # print("Away Team:", away_team['name'])

        home_team_matches = match_utils.get_team_matches(
            home_team_id, game_date, user_token, current_ground=False)
        away_team_matches = match_utils.get_team_matches(
            away_team_id, game_date, user_token, current_ground=False)

        # logger.info("Home Team Matches:")
        # logger.info(home_team_matches)
        # logger.info("Away Team Matches:")
        # logger.info(away_team_matches)

        home_team_matches_with_stats = calculate_team_stats(
            home_team_matches, home_team_id)
        away_team_matches_with_stats = calculate_team_stats(
            away_team_matches, away_team_id)

        # Ensure that there are stats available for the first match in each list
        if home_team_matches_with_stats:
            home_stats = home_team_matches_with_stats[0]
        else:
            home_stats = {}

        if away_team_matches_with_stats:
            away_stats = away_team_matches_with_stats[0]
        else:
            away_stats = {}

        set_target = 0
        if target == 'h':
            set_target = 1 if winner == 1 else 0
        elif target == 'd':
            set_target = 1 if winner == 0 else 0
        elif target == 'a':
            set_target = 1 if winner == 2 else 0
        elif target == 'o':
            set_target = 1 if goals > 2 else 0
        elif target == 'u':
            set_target = 1 if goals <= 2 else 0
        elif target == 'gg':
            set_target = 1 if gg else 0
        elif target == 'ng':
            set_target = 1 if not gg else 0

        # After calculating the statistics, create a DataFrame for the current match
        current_match_data = {
            "date": date,
            "hour": hour,
            "day_of_week": day_of_week,
            'competition_id': competition['id'],
            'season_id': season_id,
            'country_id': country_id,
            'home_team_id': home_team['id'],
            'away_team_id': away_team['id'],
            'home_team_wins': home_stats.get('team_wins', 0),
            'home_team_draws': home_stats.get('draws', 0),
            'home_team_loses': home_stats.get('team_loses', 0),
            'home_team_totals': home_stats.get('totals', 0),
            'home_team_goal_for_avg': home_stats.get('goal_for_avg', 0),
            'home_team_goal_against_avg': home_stats.get('goal_against_avg', 0),
            'away_team_wins': away_stats.get('team_wins', 0),
            'away_team_draws': away_stats.get('draws', 0),
            'away_team_loses': away_stats.get('team_loses', 0),
            'away_team_totals': away_stats.get('totals', 0),
            'away_team_goal_for_avg': away_stats.get('goal_for_avg', 0),
            'away_team_goal_against_avg': away_stats.get('goal_against_avg', 0),
            'target': set_target,
        }

        matches_data.append(current_match_data)

    return matches_data
