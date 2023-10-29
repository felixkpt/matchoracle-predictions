import requests
from config import API_BASE_URL

def get_team_matches(team_id, game_date, user_token, current_ground=False, per_page=15):
    if not user_token:
        raise ValueError("User token is required to fetch team matches.")

    # Create a dictionary with the headers
    headers = {
        "Authorization": f"Bearer {user_token}",
    }

    base_url = f"{API_BASE_URL}/admin/teams/view"
    matches_url = f"{base_url}/{team_id}/matches?type=played&per_page={per_page}&before={game_date}"

    if current_ground:
        matches_url += '&currentground=home'  # or '&currentground=away' as needed

    response = requests.get(matches_url, headers=headers)

    if response.status_code == 200:
        response_json = response.json()

        matches = response_json.get('results', {}).get('data', [])

        # Modify each match to include additional data
        for match in matches:
            # Extract scores from the match and add them to the match dictionary
            match['home_scores_full_time'] = match['score']['home_scores_full_time'] if match['score'] and 'home_scores_full_time' in match['score'] else None
            match['away_scores_full_time'] = match['score']['away_scores_full_time'] if match['score'] and 'away_scores_full_time' in match['score'] else None
            match['home_scores_half_time'] = match['score']['home_scores_half_time'] if match['score'] and 'home_scores_half_time' in match['score'] else None
            match['away_scores_half_time'] = match['score']['away_scores_half_time'] if match['score'] and 'away_scores_half_time' in match['score'] else None
            
        return matches
    else:
        return []
