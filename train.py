from run_train import run_train
from datetime import datetime
from dateutil.relativedelta import relativedelta
from configs.active_competitions.competitions_data import get_competitions, get_trained_competitions
import argparse

# Calculate from_date and to_date
TRAIN_TO_DATE = datetime.today() + relativedelta(days=-30 * 6)


def train(user_token, prediction_type=None, hyperparameters={}):
    print("\n............... START TRAIN PREDICTIONS ..................\n")

    parser = argparse.ArgumentParser(
        description='Train predictions with different configurations.')
    parser.add_argument('--competition', type=int, help='Competition ID')
    parser.add_argument('--target', choices=['hda', 'ft-hda', 'ht-hda', 'bts', 'over15', 'over25', 'over35', 'cs'],
                        help='Target for predictions')

    parser.add_argument('--ignore-saved', action='store_true',
                        help='Ignore saved data')
    parser.add_argument('--is-grid-search',
                        action='store_true', help='Enable grid search')

    parser.add_argument('--ignore-trained', action='store_true',
                        help='Ignore timing data')

    args, extra_args = parser.parse_known_args()
    target = args.target
    ignore_saved = args.ignore_saved
    is_grid_search = args.is_grid_search
    ignore_trained = args.ignore_trained

    print(f"Main Prediction Target: {target if target else 'all'}")
    print(f"Training to {TRAIN_TO_DATE}")

    # If competition_id is provided, use it; otherwise, fetch from the backend API
    competition_ids = [{"id": args.competition, "name": "N/A"}
                       ] if args.competition is not None else get_competitions(user_token)

    trained_competition_ids = [
    ] if ignore_trained or args.competition else get_trained_competitions()

    arr = []
    for compe in competition_ids:
        if not trained_competition_ids or compe['id'] not in trained_competition_ids:
            arr.append(compe)

    print(f"Competitions info:")
    print(f"Active: {len(competition_ids)}, to be trained: {len(arr)}, aleady trained: {len(competition_ids)-len(arr)}\n")

    competition_ids = arr

    # Starting points for loops
    start_from = [12, 6, 4]
    end_at = [12, 6, 4]

    for history_limit_per_match in [6, 10, 12, 15]:
        # Skip if current history_limit_per_match is less than the specified starting point
        if history_limit_per_match < start_from[0] or history_limit_per_match > end_at[0]:
            continue
        else:
            for current_ground_limit_per_match in [4, 6, 8]:
                if current_ground_limit_per_match < start_from[1] or current_ground_limit_per_match > end_at[1]:
                    continue
                else:
                    for h2h_limit_per_match in [4, 6, 8]:
                        if h2h_limit_per_match < start_from[2] or h2h_limit_per_match > end_at[2]:
                            continue
                        else:
                            # Generate prediction type based on loop parameters
                            PREDICTION_TYPE = (
                                prediction_type
                                or f"regular_prediction_{history_limit_per_match}_{current_ground_limit_per_match}_{h2h_limit_per_match}"
                            )

                            i = 0
                            # Loop over competition IDs
                            for competition in competition_ids:
                                i += 1
                                COMPETITION_ID = competition['id']

                                compe_data = {'id': COMPETITION_ID,
                                              'name': competition['name'],
                                              'prediction_type': PREDICTION_TYPE}
                                # Parameters for training
                                be_params = {
                                    'history_limit_per_match': history_limit_per_match,
                                    'current_ground_limit_per_match': current_ground_limit_per_match,
                                    'h2h_limit_per_match': h2h_limit_per_match,
                                    'to_date': TRAIN_TO_DATE,
                                }

                                print(
                                    f"{i}/{len(competition_ids)}. COMPETITION #{COMPETITION_ID}, ({competition['name']})")
                                print(
                                    f"***** START TRAIN PREDICTS FOR {COMPETITION_ID} *****")

                                # Run training for the current configuration
                                run_train(user_token, compe_data=compe_data, target=target, be_params=be_params,
                                          ignore_saved=ignore_saved, is_grid_search=is_grid_search)
                                print(
                                    f"***** END TRAIN PREDICTS FOR {COMPETITION_ID} *****\n")

                                # return 0
    print(f"\n....... END TRAIN PREDICTIONS, Happy coding! ........")
