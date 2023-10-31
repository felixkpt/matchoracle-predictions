import requests
from auth_utils import get_user_token
from datetime import datetime
import logging
from config import API_BASE_URL
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from matches.entry import get_and_do_stats

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

        target = 'o'
        # Now that you have the user token, you can use it for other API requests.
        train_url = f"{API_BASE_URL}/admin/competitions/view/48/matches?type=played&per_page=2500&from_date=2022-06-01&to_date=2023-06-01"
        test_url = f"{API_BASE_URL}/admin/competitions/view/48/matches?type=played&per_page=1000&from_date=2023-06-02&to_date=2023-10-28"
        
        train_matches_data = get_and_do_stats(url=train_url, API_BASE_URL=API_BASE_URL, user_token=user_token, game_date=game_date, target=target)

        # Create train DataFrame
        train = pd.DataFrame(train_matches_data)
        train['competition_id'] = train['competition_id'].astype('category').cat.codes
        train['season_id'] = train['season_id'].astype('category').cat.codes
        train['country_id'] = train['country_id'].astype('category').cat.codes
        train['home_team_id'] = train['home_team_id'].astype('category').cat.codes
        train['away_team_id'] = train['away_team_id'].astype('category').cat.codes

        test_matches_data = get_and_do_stats(url=test_url, API_BASE_URL=API_BASE_URL, user_token=user_token, game_date=game_date, target=target)
        
        # Create test DataFrame
        test = pd.DataFrame(test_matches_data)
        test['competition_id'] = test['competition_id'].astype('category').cat.codes
        test['season_id'] = test['season_id'].astype('category').cat.codes
        test['country_id'] = test['country_id'].astype('category').cat.codes
        test['home_team_id'] = test['home_team_id'].astype('category').cat.codes
        test['away_team_id'] = test['away_team_id'].astype('category').cat.codes

        print(train)

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
    train = pd.DataFrame(
        [match for match in data if match['date'] < cutoff_date])
    test = pd.DataFrame(
        [match for match in data if match['date'] > cutoff_date])

    # Encode categorical variables to numeric
    train['competition_id'] = train['competition_id'].astype(
        'category').cat.codes
    train['season_id'] = train['season_id'].astype('category').cat.codes
    train['country_id'] = train['country_id'].astype('category').cat.codes
    train['home_team_id'] = train['home_team_id'].astype('category').cat.codes
    train['away_team_id'] = train['away_team_id'].astype('category').cat.codes

    test['competition_id'] = test['competition_id'].astype(
        'category').cat.codes
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
