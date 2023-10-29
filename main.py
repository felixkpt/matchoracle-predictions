import requests
from auth_utils import get_user_token
from datetime import datetime
import match_utils
import logging
from config import API_BASE_URL
from team_stats import calculate_team_stats
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

rf = RandomForestClassifier(
    n_estimators=50, min_samples_split=10, random_state=1)

EMAIL = "admin@example.com"
PASSWORD = "admin@example.com"
# Example: Get team matches for a given date
game_date = datetime.now().strftime("%Y-%m-%d")


def configure_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def make_api_request(endpoint, method="GET", headers=None, data=None):
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.request(method, url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


def main():
    # Get the user token
    user_token = get_user_token(EMAIL, PASSWORD)

    # Initialize the logger
    logger = configure_logger()

    if user_token:
        logger.info("User token obtained successfully.")

        # Create a dictionary with the headers
        headers = {
            "Authorization": f"Bearer {user_token}",
        }

        # Now that you have the user token, you can use it for other API requests.
        url = f"{API_BASE_URL}/admin/competitions/view/48/matches?type=played&per_page=500"
        # Make a GET request with the headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error if the request was not successful

        matches_data = []
        # Parse the JSON response
        matches = response.json()['results']['data']

        for match in matches:

            # Now that you have the user token, you can use it for other API requests.
            url = f"{API_BASE_URL}/admin/matches/view/{match['id']}"

            # Create a dictionary with the headers
            headers = {
                "Authorization": f"Bearer {user_token}",
            }
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
                'target': home_stats.get('team_wins', 0),
            }

            matches_data.append(current_match_data)

        # Create a DataFrame
        # df = pd.DataFrame(matches_data)

        # print(df.dtypes)
        cutoff_date = pd.to_datetime('2023-05-19 15:00:00')

        train = pd.DataFrame(
            [match for match in matches_data if (match['date']) < cutoff_date])
        # train['competition_id'] = train['competition_id'].astype('category').cat.codes
        # train['season_id'] = train['season_id'].astype('category').cat.codes
        # train['country_id'] = train['country_id'].astype('category').cat.codes
        # train['home_team_id'] = train['home_team_id'].astype('category').cat.codes
        # train['away_team_id'] = train['away_team_id'].astype('category').cat.codes
        
        test = pd.DataFrame(
            [match for match in matches_data if match['date'] >= cutoff_date])
        # test['competition_id'] = test['competition_id'].astype('category').cat.codes
        # test['season_id'] = test['season_id'].astype('category').cat.codes
        # test['country_id'] = test['country_id'].astype('category').cat.codes
        # test['home_team_id'] = test['home_team_id'].astype('category').cat.codes
        # test['away_team_id'] = test['away_team_id'].astype('category').cat.codes
        
        
        # print(train)
        
        predictors = ['hour',
                      'day_of_week',
                      'competition_id',
                      'season_id',
                      'country_id',
                      'home_team_id',
                      'away_team_id',
                      'home_team_wins', 'home_team_draws', 'home_team_loses', 'home_team_totals',
                      'away_team_wins', 'away_team_draws', 'away_team_loses', 'away_team_totals',
                      ]

        rf.fit(train[predictors], train['target'])

        preds = rf.predict(test[predictors])

        accuracy = accuracy_score(test['target'], preds)

        precision = precision_score(test['target'], preds)

        print(accuracy)
        print(precision)

        # combined = pd.DataFrame(dict(actual=test['target'], predictions=preds))

        # print(pd.crosstab(index=combined['actual'], columns=combined['predictions']))

        # grouped_matches = matches.groupby('team')

        # cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
        # new_cols = [f"{c}_rolling" for c in cols]

        # group = grouped_matches.get_group('Liverpool')

        # matches_rolling = grouped_matches.apply(lambda x: rolling_averages(x, cols, new_cols ))

        # matches_rolling = matches_rolling.droplevel('team')

        # # assigning indexes from 0 to max
        # matches_rolling.index = range(matches_rolling.shape[0])

        # # print(matches_rolling)

        # combined, precision = make_predictions(matches_rolling, predictors + new_cols)

        # print(precision)

    else:
        logger.error(
            "Failed to obtain the user token. Check your credentials or the login process.")


def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


def make_predictions(data, predictors, cutoff_date):
    
    # Create DataFrames from the list of dictionaries
    train = pd.DataFrame([match for match in data if match['date'] < cutoff_date])
    test = pd.DataFrame([match for match in data if match['date'] > cutoff_date])

    # Encode categorical variables to numeric
    train['competition_id'] = train['competition_id'].astype('category').cat.codes
    train['season_id'] = train['season_id'].astype('category').cat.codes
    train['country_id'] = train['country_id'].astype('category').cat.codes
    train['home_team_id'] = train['home_team_id'].astype('category').cat.codes
    train['away_team_id'] = train['away_team_id'].astype('category').cat.codes

    test['competition_id'] = test['competition_id'].astype('category').cat.codes
    test['season_id'] = test['season_id'].astype('category').cat.codes
    test['country_id'] = test['country_id'].astype('category').cat.codes
    test['home_team_id'] = test['home_team_id'].astype('category').cat.codes
    test['away_team_id'] = test['away_team_id'].astype('category').cat.codes

    # Define the predictors
    predictors = ['hour', 'day_of_week', 'competition_id', 'season_id', 'country_id',
                  'home_team_id', 'away_team_id', 'home_team_wins', 'home_team_draws', 'home_team_loses', 'home_team_totals',
                  'away_team_wins', 'away_team_draws', 'away_team_loses', 'away_team_totals']

    # Fit the RandomForestClassifier
    rf.fit(train[predictors], train['target'])

    # Make predictions
    preds = rf.predict(test[predictors])

    # Calculate accuracy and precision
    accuracy = accuracy_score(test['target'], preds)
    precision = precision_score(test['target'], preds)

    return accuracy, precision

if __name__ == "__main__":
    main()
