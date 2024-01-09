import argparse
from run_train import run_train
from datetime import datetime
from configs.active_competitions.competition_data import get_competition_ids, get_trained_competitions

# Define constants
from configs.settings import HISTORY_LIMITS

# Define constants
TRAIN_FROM_DATE = datetime.strptime('2015-01-01', '%Y-%m-%d')
TRAIN_TO_DATE = datetime.strptime('2023-01-01', '%Y-%m-%d')


def train(user_token, target=None, prediction_type=None, hyperparameters={}):
    print("\n............... START TRAIN PREDICTIONS ..................\n")

    parser = argparse.ArgumentParser(
        description='Train predictions with different configurations.')
    parser.add_argument('--competition', type=int, help='Competition ID')
    parser.add_argument('--ignore-saved', action='store_true',
                        help='Ignore saved models')

    args, extra_args = parser.parse_known_args()

    ignore_saved = args.ignore_saved
    # If competition_id is provided, use it; otherwise, fetch from the backend API
    competition_ids = [
        args.competition] if args.competition is not None else get_competition_ids(user_token)

    if not ignore_saved:
        competition_ids = [
            c for c in competition_ids if c not in get_trained_competitions(True)]

    # Set default prediction type or use provided prediction_type
    # If prediction_type is provided, restrict history_limits to [8]
    HISTORY_LIMITS = [7, 10]
    history_limits = HISTORY_LIMITS
    if prediction_type:
        history_limits = [8]

    # Starting and ending points for loops
    start_from = [10, 5, 5]
    end_at = [10, 5, 5]

    for history_limit_per_match in history_limits:
        # Skip if current history_limit_per_match is outside the specified range
        if not (start_from[0] <= history_limit_per_match <= end_at[0]):
            continue
        else:
            for current_ground_limit_per_match in [5]:
                if current_ground_limit_per_match > history_limit_per_match or current_ground_limit_per_match + 1 >= history_limit_per_match:
                    continue

                # Skip if current_ground_limit_per_match is outside the specified range
                if not (start_from[1] <= current_ground_limit_per_match <= end_at[1]):
                    continue
                else:
                    for h2h_limit_per_match in [5]:
                        if h2h_limit_per_match > current_ground_limit_per_match:
                            continue

                        # Skip if h2h_limit_per_match is outside the specified range
                        if not (start_from[2] <= h2h_limit_per_match <= end_at[2]):
                            continue
                        else:
                            # Generate prediction type based on loop parameters
                            PREDICTION_TYPE = (
                                prediction_type
                                or f"regular_prediction_{history_limit_per_match}_{current_ground_limit_per_match}_{h2h_limit_per_match}"
                            )

                            # Loop over competition IDs
                            for COMPETITION_ID in competition_ids:
                                # Parameters for training
                                be_params = {
                                    'history_limit_per_match': history_limit_per_match,
                                    'current_ground_limit_per_match': current_ground_limit_per_match,
                                    'h2h_limit_per_match': h2h_limit_per_match
                                }

                                # Run training for the current configuration
                                run_train(user_token, target=target, hyperparameters=hyperparameters, PREDICTION_TYPE=PREDICTION_TYPE,
                                          COMPETITION_ID=COMPETITION_ID, TRAIN_TO_DATE=TRAIN_TO_DATE, TRAIN_FROM_DATE=TRAIN_FROM_DATE, be_params=be_params)
                                # return 0

    print(f"\n....... END TRAIN PREDICTIONS, Happy coding! ........")
