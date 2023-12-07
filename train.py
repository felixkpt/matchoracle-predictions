from run_train import run_train
from datetime import datetime

# Define constants
from configs.settings import HISTORY_LIMITS
# Competitions to consider: Campeonato Brasileiro Série A, Championship, EPL, Portugal primera, LaLiga
# Corresponding Competition IDs: 25, 47, 48, 125, 148
COMPETITION_IDS = [25, 47, 48, 125, 148]
# Example configurations for testing:
# COMPETITION_IDS = [25, 47, 48]
COMPETITION_IDS = [25, 47, 48, 125, 148]

# Calculate from_date and to_date
TRAIN_TO_DATE = datetime.strptime('2023-08-31', '%Y-%m-%d')


def train(user_token, target=None, prediction_type=None, hyperparameters={}):
    print("\n............... START TRAIN PREDICTIONS ..................\n")

    # Set default prediction type or use provided prediction_type
    # If prediction_type is provided, restrict history_limits to [8]
    HISTORY_LIMITS = [7, 10, 12, 15]
    history_limits = HISTORY_LIMITS
    if prediction_type:
        history_limits = [8]

    # Starting points for loops
    start_from = [0, 0, 0]

    for history_limit_per_match in history_limits:
        # Skip if current history_limit_per_match is less than the specified starting point
        if history_limit_per_match < start_from[0]:
            continue
        else:
            for current_ground_limit_per_match in [4, 6, 8]:
                # Skip if current_ground_limit_per_match is less than the specified starting point
                if (
                    history_limit_per_match == start_from[0]
                    and current_ground_limit_per_match < start_from[1]
                ):
                    continue
                else:
                    for h2h_limit_per_match in [4, 6, 8]:
                        # Skip if h2h_limit_per_match is less than the specified starting point
                        if (
                            current_ground_limit_per_match == start_from[1]
                            and h2h_limit_per_match < start_from[2]
                        ):
                            continue
                        else:
                            # Generate prediction type based on loop parameters
                            PREDICTION_TYPE = (
                                prediction_type
                                or f"regular_prediction_{history_limit_per_match}_{current_ground_limit_per_match}_{h2h_limit_per_match}"
                            )

                            # Loop over competition IDs
                            for COMPETITION_ID in COMPETITION_IDS:
                                # Parameters for training
                                be_params = {
                                    'history_limit_per_match': history_limit_per_match,
                                    'current_ground_limit_per_match': current_ground_limit_per_match,
                                    'h2h_limit_per_match': h2h_limit_per_match
                                }

                                # Run training for the current configuration
                                run_train(user_token, target=target, hyperparameters=hyperparameters, PREDICTION_TYPE=PREDICTION_TYPE,
                                          COMPETITION_ID=COMPETITION_ID, TRAIN_TO_DATE=TRAIN_TO_DATE, be_params=be_params)
                                # return 0

    print(f"\n....... END TRAIN PREDICTIONS, Happy coding! ........")
